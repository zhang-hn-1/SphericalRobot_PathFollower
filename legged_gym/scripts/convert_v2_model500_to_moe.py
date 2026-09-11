"""Convert the verified V2 model_500 actor into three identical MOE heads."""
from pathlib import Path
import torch

ROOT = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911")
SOURCE = ROOT / "logs/rotunbot_path/Sep11_01-53-15_geometric_path_v2_from_scratch/model_500.pt"
TARGET_DIR = ROOT / "logs/rotunbot_path/v4_moe_seed_from_v2_model500"
TARGET = TARGET_DIR / "model_500.pt"
checkpoint = torch.load(str(SOURCE), map_location="cpu")
old_state = checkpoint["model_state_dict"]
new_state = {}
actor_keys = 0
for key, value in old_state.items():
    if key.startswith("actor."):
        suffix = key[len("actor."):]
        for expert in ("straight", "left", "right"):
            new_state[f"actors.{expert}.{suffix}"] = value.clone()
        actor_keys += 1
    else:
        new_state[key] = value
if actor_keys == 0:
    raise RuntimeError("Source checkpoint has no actor.* parameters")
checkpoint["model_state_dict"] = new_state
checkpoint["infos"] = {
    "conversion": "duplicate V2 model_500 actor into straight/left/right heads",
    "source": str(SOURCE),
    "optimizer_must_be_reset": True,
}
TARGET_DIR.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, str(TARGET))
print({"source": str(SOURCE), "target": str(TARGET), "actor_parameter_tensors_copied": actor_keys})