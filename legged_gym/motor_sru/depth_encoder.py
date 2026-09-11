"""Pretrained SRU visual front-end.

The encoder and spatial attention in this module follow the official SRU
navigation repositories.  The depth VAE is used only as a frozen feature
extractor; PPO trains the attention layers that fuse those features with the
robot and goal observations.

Encoder source:
    sru-navigation-sim/isaaclab_nav_task/navigation/mdp/depth_noise_encoder.py
Attention source:
    sru-navigation-learning/rsl_rl/networks/sru_memory/attention.py
"""

import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import regnet_x_400mf
    from torchvision.ops import FeaturePyramidNetwork
except ImportError as error:
    raise ImportError(
        "ActorCriticDepth's pretrained SRU encoder requires a torchvision "
        "version that provides regnet_x_400mf and FeaturePyramidNetwork."
    ) from error

# torchvision exported this helper from torchvision.ops.misc before exposing
# it from torchvision.ops.  Isaac Gym environments commonly use that older
# torchvision release, so support both locations without upgrading PyTorch.
try:
    from torchvision.ops import Conv2dNormActivation
except ImportError:
    try:
        from torchvision.ops.misc import Conv2dNormActivation
    except ImportError:
        class Conv2dNormActivation(nn.Sequential):
            """Backport of torchvision's Conv2d-Norm-Activation helper.

            The old Isaac Gym torchvision build does not expose this utility.
            Keeping the same Sequential layer order preserves all checkpoint
            parameter names (``0.*`` for conv and ``1.*`` for batch norm).
            """

            def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=None,
                groups=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                dilation=1,
                inplace=True,
                bias=None,
            ):
                if padding is None:
                    padding = (kernel_size - 1) // 2 * dilation
                if bias is None:
                    bias = norm_layer is None
                layers = [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                    )
                ]
                if norm_layer is not None:
                    layers.append(norm_layer(out_channels))
                if activation_layer is not None:
                    try:
                        layers.append(activation_layer(inplace=inplace))
                    except TypeError:
                        layers.append(activation_layer())
                super().__init__(*layers)
                self.out_channels = out_channels


class VAESampler(nn.Module):
    """Official deterministic VAE sampler (the latent mean is returned)."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv = Conv2dNormActivation(
            input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(
                latent_dim,
                latent_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, inputs):
        return self.mean_layers(self.conv(inputs))


class DepthEncoder(nn.Module):
    """Official RegNet-X-400MF feature pyramid depth encoder."""

    def __init__(self, out_channels):
        super().__init__()
        try:
            encoder = regnet_x_400mf(weights=None)
        except TypeError:
            # torchvision before the multi-weight API used pretrained=False.
            encoder = regnet_x_400mf(pretrained=False)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        encoder[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channels)

    def forward(self, depth):
        features = OrderedDict()
        depth = self.enc(depth)
        features["feat1"] = self.enc_1(depth)
        features["feat2"] = self.enc_2(features["feat1"])
        features["feat3"] = self.enc_3(features["feat2"])
        return self.fpn(features)["feat1"]


class DepthDecoder(nn.Module):
    """Decoder kept so the official full VAE checkpoint loads strictly."""

    def __init__(self, input_dim):
        super().__init__()
        self.conv = Conv2dNormActivation(
            input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, latent):
        return self.decoder(self.conv(latent))


class DepthVAENet(nn.Module):
    """Checkpoint-compatible official depth VAE."""

    def __init__(self, latent_dim):
        super().__init__()
        self.depth_encoder = DepthEncoder(latent_dim)
        self.vae_sampler = VAESampler(latent_dim, latent_dim)
        self.depth_decoder = DepthDecoder(latent_dim)

    def forward(self, depth):
        return self.vae_sampler(self.depth_encoder(depth))


class FrozenDepthFeatureEncoder(nn.Module):
    """Load the official VAE weights and permanently freeze the feature path."""

    @staticmethod
    def _remap_fpn_keys_for_torchvision(state_dict, model_state_dict):
        """Translate FPN keys between old and new torchvision layouts.

        New torchvision wraps each FPN convolution in a one-element Sequential,
        producing ``inner_blocks.0.0.weight``.  Older Isaac Gym builds store the
        same convolution directly as ``inner_blocks.0.weight``.  Only rename a
        key when the candidate is present in the model's own state dict, then
        retain strict loading to catch every real architecture mismatch.
        """

        model_keys = set(model_state_dict.keys())
        remapped = OrderedDict()
        remap_count = 0
        block_names = ("inner_blocks", "layer_blocks")
        for key, value in state_dict.items():
            target_key = key
            if key not in model_keys:
                for block_name in block_names:
                    marker = ".fpn." + block_name + "."
                    if marker not in key:
                        continue
                    prefix, suffix = key.split(marker, 1)
                    parts = suffix.split(".")
                    candidates = []
                    if len(parts) >= 3 and parts[1] == "0":
                        candidates.append(
                            prefix + marker + ".".join((parts[0],) + tuple(parts[2:]))
                        )
                    if len(parts) >= 2 and parts[1] != "0":
                        candidates.append(
                            prefix + marker + ".".join((parts[0], "0") + tuple(parts[1:]))
                        )
                    matched = next(
                        (candidate for candidate in candidates if candidate in model_keys),
                        None,
                    )
                    if matched is not None:
                        target_key = matched
                        remap_count += 1
                        break
            if target_key in remapped:
                raise RuntimeError("Duplicate checkpoint key after FPN remapping: " + target_key)
            remapped[target_key] = value
        return remapped, remap_count

    def __init__(self, latent_dim, checkpoint_path):
        super().__init__()
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Pretrained SRU depth encoder checkpoint was not found: "
                + checkpoint_path
            )
        self.model = DepthVAENet(latent_dim)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Pretrained SRU depth checkpoint is not a state dict")
        checkpoint, remap_count = self._remap_fpn_keys_for_torchvision(
            checkpoint, self.model.state_dict()
        )
        self.model.load_state_dict(checkpoint, strict=True)
        if remap_count:
            print(
                "Remapped {} pretrained FPN keys for this torchvision version".format(
                    remap_count
                )
            )
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        # PPO toggles the complete actor-critic between train/eval.  BatchNorm in
        # the frozen encoder must continue using its pretrained running moments.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, depth):
        with torch.no_grad():
            return self.model(depth)


