"""Decide why sharp S-curves fail: preview phase error or curvature authority.

s_curve at |k| = 0.40-0.45 is the only remaining weak bucket, its failures are
`deviation` (leaving the 1.5 m corridor), and it sits in a specific band of the
prior.  Two mechanisms could produce that, and they need opposite fixes:

1. **Preview phase error.**  For S-curves the curvature is a sine whose
   wavelength equals the path length, and the steering feedforward uses the
   curvature at ``path_index + lookahead``.  The low and mid S bands use a fixed
   ``prior_s_lookahead`` of 1.20 m, which on a 3-5 m path is 96-144 degrees of
   phase - close to antiphase.  The tight band (peak >= 0.45) instead uses a
   length-scaled lookahead of 20% of the path, which is a constant 72 degrees.
   If the preview is the problem, the sign of the feedforward curvature will
   disagree with the curvature the ball actually needs, often.

2. **Curvature authority.**  ``|k|max ~= 0.092 + 0.92*|a1|`` was identified
   open loop.  The mid band drives at ``prior_s_mid_drive = 0.40``, which allows
   about 0.46 1/m against the 0.40 1/m required - a 15% margin before the
   cross-track and heading corrections are added.  If authority is the problem,
   the joint-2 command will sit pinned at its limit.

The fix for (1) is a lookahead change; for (2) it is slowing down on S-curves,
which trades against the 40 s episode budget.  This script separates them by
logging, per step, the feedforward curvature against the curvature at the
ball's current path index, plus how often the joint-2 command saturates.

    PATH_DIAG_CURVATURE=0.40 ./run_local_script.sh \
        legged_gym/scripts/diagnose_s_curve_steering.py
"""

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacgym  # noqa: F401  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

TYPE_LOOKUP = {"straight": 0, "left_arc": 1, "right_arc": 2, "s_curve": 3}
SATURATION_EPS = 0.999


def main():
    args = get_args()
    seed = int(os.environ.get("PATH_DIAG_SEED", "4200"))
    stage = int(os.environ.get("PATH_DIAG_STAGE", "6"))
    type_name = os.environ.get("PATH_DIAG_TYPE", "s_curve")
    curvature = float(os.environ.get("PATH_DIAG_CURVATURE", "0.40"))
    episodes = int(os.environ.get("PATH_DIAG_ENVS", "128"))

    env_cfg, train_cfg = task_registry.get_cfgs(name="rotunbot_path")
    if not bool(getattr(env_cfg.control, "use_path_action_prior", False)):
        raise SystemExit("PATH_USE_ACTION_PRIOR=1 is required")
    if not bool(getattr(train_cfg.policy, "zero_init_actor_output", False)):
        raise SystemExit("PATH_ZERO_INIT_ACTOR=1 is required")

    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg.env.num_envs = episodes
    env_cfg.noise.add_noise = False
    env_cfg.path.curriculum_enabled = False
    env_cfg.domain_rand.push_robots = False
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name="rotunbot_path", args=args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)
    probe = env.get_observations()
    with torch.no_grad():
        if float(policy(probe).abs().max().item()) != 0.0:
            raise SystemExit("actor output is not zero; the action would not equal the prior")

    env.cfg.path.curriculum_enabled = False
    env.path_curriculum_stage = stage
    env.forced_path_type = TYPE_LOOKUP[type_name]
    env.forced_curvature = curvature
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()

    batch = torch.arange(env.num_envs, device=env.device)
    zero_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    rows = []
    with torch.no_grad():
        for step in range(int(env.max_episode_length * 2)):
            reference = env._lookahead_curvature()          # feedforward curvature
            current = env.path_curvature[batch, env.path_index]  # needed right now
            prior = env._analytic_action_prior()
            phase = env.path_s / torch.clamp(env.path_length, min=1e-6)
            rows.append(torch.stack([
                phase,
                reference,
                current,
                prior[:, 1].abs(),                          # joint-2 command magnitude
                env.path_cross_track.abs(),
                env.path_remaining,
            ], dim=1).cpu().numpy())
            if bool(env.step(zero_action)[3].all()):
                break

    data = np.concatenate(rows, axis=0)
    columns = ["phase_fraction", "reference_curvature", "current_curvature",
               "abs_joint2_command", "abs_cross_track_m", "remaining_m"]
    remaining = data[:, 5]
    # Only the moving part of the episode is interesting; past the endpoint the
    # ball is parking and the curvature question no longer applies.
    moving = remaining > 0.10
    sub = data[moving]
    ref, cur = sub[:, 1], sub[:, 2]

    # Sign disagreement is the phase-error signature: a feedforward sampled close
    # to antiphase points the wrong way.  Weight by magnitude so the many
    # near-zero crossings do not dominate.
    eps = 0.02
    both = (np.abs(ref) > eps) & (np.abs(cur) > eps)
    disagree = float(np.mean(np.sign(ref[both]) != np.sign(cur[both]))) if both.any() else float("nan")
    corr = float(np.corrcoef(ref[both], cur[both])[0, 1]) if both.sum() > 2 else float("nan")
    saturation = float(np.mean(sub[:, 3] >= SATURATION_EPS))

    summary = {
        "path_type": type_name, "curvature": curvature, "episodes": episodes,
        "steps_considered": int(len(sub)),
        "sign_disagreement_rate": disagree,
        "correlation_reference_vs_current": corr,
        "joint2_saturation_rate": saturation,
        "abs_joint2_command_median": float(np.median(sub[:, 3])),
    }

    out = Path(os.environ.get(
        "PATH_DIAG_OUTPUT",
        str(PROJECT_ROOT / "artifacts" / "path_follower" /
            f"diagnose_{type_name}_k{curvature:+.3f}".replace(".", "p"))))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(data.tolist())

    print(f"{type_name} k={curvature:+.3f}  {episodes} envs  {len(sub)} steps while moving")
    print(f"  sign(reference) != sign(current) : {disagree:6.1%}   <- phase-error signature")
    print(f"  corr(reference, current)         : {corr:6.3f}")
    print(f"  joint-2 command pinned at limit  : {saturation:6.1%}   <- authority signature")
    print(f"  median |joint-2 command|         : {summary['abs_joint2_command_median']:.3f}")
    print()
    print(f"{'phase':>10}{'n':>8}{'ref k':>9}{'current k':>11}{'|j2|':>8}{'xtrack':>8}")
    for low in np.arange(0.0, 1.0, 0.1):
        mask = (sub[:, 0] >= low) & (sub[:, 0] < low + 0.1)
        if mask.sum() < 5:
            continue
        block = sub[mask]
        print(f"{low:>4.1f}-{low + 0.1:<5.1f}{int(mask.sum()):>8}"
              f"{np.median(block[:, 1]):>9.3f}{np.median(block[:, 2]):>11.3f}"
              f"{np.median(block[:, 3]):>8.3f}{np.median(block[:, 4]):>8.3f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
