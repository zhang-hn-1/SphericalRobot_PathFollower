# Running this checkout on a local machine

The project was developed on a server with the paths `/data/lzq/workspace/SphericalRobot_PathFollower_20260911`
and `/data/lzq/conda_envs/isaacgym/bin/python`.  Every archived `launch_*.sh`,
`eval_*.sh` and `scan_*.sh` script hardcodes those paths, so they cannot run
here.  This document records what this machine needs and how to run things.

## Interpreters and paths

| Item | Value on this machine |
|---|---|
| Interpreter | `/home/jason/legged_gym/.venv/bin/python` (Python 3.8.10, uv venv) |
| torch | 1.10.0+cu113 |
| IsaacGym | Preview 4, editable install from `/home/jason/文档/xwechat_files/.../IsaacGym_Preview_4_Package` |
| GPU | RTX 4070 Laptop, 8 GB, driver 535 / CUDA 12.2 |

`legged_gym` is also installed editable into that shared venv, pointing at a
**different checkout** (`/home/jason/legged_gym/legged_gym`).  Nothing here sets
`PYTHONPATH`, so a bare `python legged_gym/scripts/train_path.py` silently
imports the other project's sources.  The launchers below put this checkout
first on `PYTHONPATH` to prevent that.

The venv is shared with other projects and is **not modified** by this checkout.

## Launchers

| Script | Purpose |
|---|---|
| `local_env.sh` | shared setup, sourced by the two below |
| `run_local.sh` | training; all experiment settings come from `PATH_*` env vars |
| `run_local_eval.sh` | deterministic evaluation of one checkpoint |
| `run_ablation_prior_vs_ppo.sh` | the prior-vs-learned-residual ablation |

Overridable before invoking: `PATH_FOLLOWER_PYTHON` (interpreter) and
`CUDA_VISIBLE_DEVICES` (GPU, default 0).

These files live on an exFAT volume, which does not store Unix permission bits.
If `./run_local.sh` reports "Permission denied", run `bash run_local.sh`
instead.

## Four environment blockers, and how they are handled

All four are environmental, not bugs in this project's code.  They are worked
around from inside the checkout so that no installed package changes.

1. **`numpy >= 1.24` removed the `np.float` alias.**  IsaacGym's
   `isaacgym/torch_utils.py` evaluates `dtype=np.float` as a default argument at
   import time, so importing IsaacGym aborts with
   `AttributeError: module 'numpy' has no attribute 'float'`.
   The venv has numpy 1.24.4.  `compat/sitecustomize.py` restores the removed
   aliases.  The alternative fix — pinning numpy < 1.24 — was rejected because
   the venv is shared.

2. **`ninja` must be on `PATH`.**  Torch shells out to `ninja` to build the
   `gymtorch` extension.  The venv provides `bin/ninja` but is never activated,
   so `local_env.sh` prepends that directory to `PATH`.  The compiled
   `gymtorch.so` is already cached in `~/.cache/torch_extensions/py38_cu113`, so
   `ninja` reports "no work to do" and no `nvcc` is required.

3. **`distutils.version` is unreachable.**  torch 1.10's tensorboard shim does
   `import distutils` and then reads `distutils.version.LooseVersion`, relying
   on something else having imported the submodule.  With setuptools 75 nothing
   does, and setuptools' own `_distutils` shadows the stdlib copy.
   `compat/sitecustomize.py` imports `distutils.version` at interpreter start.

4. **The CUDA 11.3 nvrtc has no `sm_89` target.**  On an Ada GPU, calling
   IsaacGym's `@torch.jit.script` helpers raises
   `nvrtc: error: invalid value for --gpu-architecture (-arch)`.  Only
   `isaacgym/torch_utils.py` is affected.  `compat/isaacgym_ada_compat.py`
   installs an import hook that re-executes that module's source with
   `torch.jit.script` as a no-op and swaps the resulting `ScriptFunction`
   attributes for the eager Python functions.  The math is identical; TorchScript
   fusion is lost in those helpers.

`compat/sitecustomize.py` is loaded automatically because `local_env.sh` puts
`compat/` on `PYTHONPATH`; CPython imports `sitecustomize` at start-up.

## Measured throughput on this machine

2048 environments, 2 iterations: 173k steps/s, 1.13-1.23 s per iteration, peak
6.9 GB of 8.2 GB VRAM.  Round numbers: 200 iterations ~4 minutes, 5000
iterations ~100 minutes.

## Two source directories exist

`legged_gym/envs/rotunbot/path_tracking/` is the **registered** environment
(`legged_gym/envs/__init__.py` imports it).  `legged_gym/envs/rotunbot_path/` is
an older fork of the same files with fewer features.  A training launch was once
wasted by editing the stale copy.  Edit only the former;
`legged_gym/tests/test_path_env_contract.py` fails if registration moves.

## Generalisation split

`PATH_LAYOUT_SPLIT=train` (default) reproduces the historical behaviour: each
curriculum stage draws curvature from its `curvature_values_stageN` list.

`PATH_LAYOUT_SPLIT=test` substitutes `heldout_curvature_values` — currently
`[0.15, 0.30, 0.45]`, which appear in **no** stage list.  `0.15` extrapolates
below the trained range; `0.30` and `0.45` interpolate between trained values.
A policy evaluated under `test` is therefore measured on curvature it never saw.
The contract test asserts the two sets stay disjoint.

## Not present on this machine

The NeuPAN source, the official Ackermann reproduction
(`/data/lzq/workspace/NeuPAN_official_repro_20260910`) and its conda environment
do not exist here; only exported artifacts under
`artifacts/path_follower/neupan_*` are available.  Curriculum stages 5 and 6 as
described in the design plan (NeuPAN-supplied paths, online replanning) are not
implemented in `rotunbot_path.py` either: the environment generates its paths
internally only.
