"""Contract checks for the rotunbot geometric path environment.

Pure stdlib: parses the registered sources instead of importing them, so this
runs without Isaac Gym, CUDA, or a GPU.  Every assertion below guards a defect
that actually shipped at some point:

* the registered task must be the directory holding the current environment.
  The tree also contains ``legged_gym/envs/rotunbot_path/``, an older fork of the
  same files; a launch was once wasted editing that copy.
* ``num_single_obs`` is 19 but the adjacent comment claimed 18-D after the V3b
  experiment was reverted in V4.
* joint 2 is limited by the URDF to 0.5236 rad; the legacy 0.45 rad hard clip
  must not come back.
* the generalisation split is only meaningful while the held-out curvature
  values are disjoint from every stage list.
* the evaluator must record (path type x curvature), otherwise the hardest
  curvature hides inside a passing path-type mean.

Run: python3 legged_gym/tests/test_path_env_contract.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE_ENV_DIR = ROOT / "legged_gym" / "envs" / "rotunbot" / "path_tracking"
STALE_ENV_DIR = ROOT / "legged_gym" / "envs" / "rotunbot_path"
CONFIG = LIVE_ENV_DIR / "rotunbot_path_config.py"
ENV = LIVE_ENV_DIR / "rotunbot_path.py"
REGISTRY = ROOT / "legged_gym" / "envs" / "__init__.py"
EVALUATOR = ROOT / "legged_gym" / "scripts" / "evaluate_path_follower.py"

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def class_assignments(path, class_name, nested=None):
    """Collect simple ``name = literal`` assignments from a (nested) class body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if nested is not None:
        outer = next(
            node for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = outer.body
        class_name = nested
    target = next(
        node for node in body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    values = {}
    for node in target.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


def test_registered_environment_is_the_live_copy():
    registry = REGISTRY.read_text(encoding="utf-8")
    check(
        "from .rotunbot.path_tracking.rotunbot_path import RotunbotPath" in registry,
        "rotunbot_path is no longer registered from .rotunbot.path_tracking; "
        "the stale legged_gym/envs/rotunbot_path/ copy may have been wired up",
    )
    check(
        "heldout_curvature_values" in CONFIG.read_text(encoding="utf-8"),
        "the registered config lacks the generalisation-split parameters; "
        "the stale duplicate was probably edited instead",
    )


def test_observation_contract():
    env_cfg = class_assignments(CONFIG, "RotunbotPathCfg", nested="env")
    single = env_cfg["num_single_obs"]
    frames = env_cfg["frame_stack"]
    short = env_cfg["short_frame_stack"]
    points = env_cfg["num_path_points"]
    point_dim = env_cfg["path_point_dim"]

    check(single == 19, f"num_single_obs changed to {single}, expected 19")
    check(frames == 20, f"frame_stack changed to {frames}, expected 20")
    check(short == 5, f"short_frame_stack changed to {short}, expected 5")
    check(
        (points, point_dim) == (10, 4),
        f"path preview changed to {points}x{point_dim}, expected 10x4",
    )

    # num_path_obs = num_path_points * path_point_dim + 3 + optional curvature
    no_curvature = points * point_dim + 3
    check(
        no_curvature == 43,
        f"path observation without lookahead curvature is {no_curvature}, expected 43",
    )
    # Actor input with the default (no curvature) configuration.
    actor_input = short * single + 16 + (points * point_dim)
    check(
        actor_input == 151,
        f"actor input width is {actor_input}, expected 151 (95 history + 16 latent + 40 preview)",
    )

    text = CONFIG.read_text(encoding="utf-8")
    check(
        "18-D" not in text,
        "config comment claims an 18-D state again; the value is 19 and the "
        "18-D wording is a leftover from the reverted V3b experiment",
    )


def test_action_and_joint_limits():
    env_cfg = class_assignments(CONFIG, "RotunbotPathCfg", nested="env")
    control = class_assignments(CONFIG, "RotunbotPathCfg", nested="control")
    limit = control["second_pos_limits"]
    mechanical = control["second_mechanical_pos_limit"]
    scale = control["second_actionScale"]
    check(
        abs(limit - 0.5236) < 1e-9 and abs(mechanical - 0.5236) < 1e-9,
        f"joint-2 limit is {limit}/{mechanical}, expected the URDF value 0.5236",
    )
    check(
        abs(scale - 0.5236) < 1e-9,
        f"joint-2 action scale is {scale}, expected 0.5236",
    )
    check(
        control["first_actionScale"] == 3.0 and control["first_vel_limits"] == 3.0,
        "joint-1 velocity target limits changed from +/-3 rad/s",
    )
    check(
        control["decimation"] == 1,
        "decimation changed; the 50 Hz policy rate assumed 1",
    )
    check(env_cfg["num_actions"] == 2, "num_actions must stay 2 (velocity, angle)")


def test_success_criteria():
    path_cfg = class_assignments(CONFIG, "RotunbotPathCfg", nested="path")
    check(
        abs(path_cfg["success_endpoint_distance"] - 0.20) < 1e-9,
        "success_endpoint_distance changed; results are not comparable across "
        "a threshold change",
    )
    check(
        abs(path_cfg["success_remaining_length"] - 0.20) < 1e-9,
        "success_remaining_length changed",
    )
    check(
        abs(path_cfg["success_speed"] - 0.10) < 1e-9,
        "success_speed changed",
    )
    check(
        path_cfg["curriculum_max_stage"] == 6,
        "curriculum_max_stage changed",
    )


def test_generalisation_split_is_disjoint():
    path_cfg = class_assignments(CONFIG, "RotunbotPathCfg", nested="path")
    trained = set()
    for stage in range(1, 7):
        for key in (f"curvature_values_stage{stage}", f"s_curvature_values_stage{stage}"):
            trained.update(path_cfg.get(key, []))

    heldout = set(path_cfg.get("heldout_curvature_values", []))
    heldout_s = set(path_cfg.get("heldout_s_curvature_values", []))
    overlap = (heldout | heldout_s) & trained
    check(
        not overlap,
        f"held-out curvature {sorted(overlap)} also appears in a training stage "
        f"list {sorted(trained)}; PATH_LAYOUT_SPLIT=test would not be held out",
    )
    check(bool(heldout), "heldout_curvature_values is empty")


def test_failure_taxonomy_is_recorded():
    env_text = ENV.read_text(encoding="utf-8")
    check(
        "path_curvature_value" in env_text,
        "the environment no longer records the per-episode path curvature",
    )
    check(
        "terminal_path_curvature" in env_text,
        "terminal_path_curvature is not populated; evaluations cannot break "
        "success down by curvature",
    )

    evaluator = EVALUATOR.read_text(encoding="utf-8")
    check(
        "by_type_curvature" in evaluator,
        "the evaluator no longer reports a (path type x curvature) breakdown",
    )
    check(
        "failure_reasons" in evaluator,
        "the evaluator no longer reports failure reasons",
    )
    check(
        "/data/lzq" not in evaluator,
        "the evaluator still hardcodes the original workstation path",
    )


def main():
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("path environment contract: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
