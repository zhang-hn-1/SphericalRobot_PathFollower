"""Promote the verified V4 model_600 to Stage 2 with lower exploration noise."""
from pathlib import Path
import torch

ROOT = Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911")
SOURCE = ROOT / "logs/rotunbot_path/Sep11_02-59-28_geometric_path_v4_moe_continual/model_600.pt"
TARGET_DIR = ROOT / "logs/rotunbot_path/v4b_stage2_seed_from_verified_model600"
TARGET = TARGET_DIR / "model_600.pt"
checkpoint = torch.load(str(SOURCE), map_location="cpu")
checkpoint["model_state_dict"]["std"].fill_(0.05)
env_state = checkpoint.setdefault("env_state", {})
env_state["path_curriculum_stage"] = 2
env_state["path_curriculum_attempts"] = 0
env_state["path_curriculum_successes"] = 0
env_state["path_curriculum_last_rate"] = 1.0
env_state["path_curriculum_type_attempts"] = [0, 0, 0, 0]
env_state["path_curriculum_type_successes"] = [0, 0, 0, 0]
env_state["path_curriculum_last_rates"] = [1.0, 1.0, 1.0, 0.0]
checkpoint["infos"] = {
    "source": str(SOURCE),
    "promotion": "Stage 1 deterministic gates all 128/128; enter Stage 2",
    "action_std": 0.05,
    "optimizer_must_be_reset": True,
}
TARGET_DIR.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, str(TARGET))
print({"source": str(SOURCE), "target": str(TARGET), "stage": 2, "std": checkpoint["model_state_dict"]["std"].tolist()})