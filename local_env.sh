#!/usr/bin/env bash
# Shared environment for running this checkout on a local machine.
#
# Sourced by run_local.sh (training) and run_local_eval.sh (evaluation); not
# meant to be executed directly.  Sets ROOT, PYTHON, PYTHONPATH, CUDA_VISIBLE_DEVICES.
#
# The archived launch_*.sh / eval_*.sh scripts hardcode the original
# workstation paths (/data/lzq/...); these wrappers derive everything from their
# own location instead.
#
# Overridable before sourcing:
#   PATH_FOLLOWER_PYTHON   interpreter to use (default: /home/jason/legged_gym/.venv/bin/python)
#   CUDA_VISIBLE_DEVICES   GPU selection      (default: 0)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PATH_FOLLOWER_PYTHON:-/home/jason/legged_gym/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "local_env.sh: interpreter not found or not executable: $PYTHON" >&2
  echo "local_env.sh: set PATH_FOLLOWER_PYTHON to the environment's python." >&2
  return 1 2>/dev/null || exit 1
fi

cd "$ROOT"
# Torch shells out to 'ninja' to build the gymtorch extension; the venv provides
# one but is not activated, so put its bin directory on PATH.
export PATH="$(dirname "$PYTHON"):$PATH"
# $ROOT must precede site-packages: 'legged_gym' is also installed editable into
# the shared venv from a different checkout, and without this the wrong sources
# would be imported silently.  compat/ carries the shims described in
# LOCAL_SETUP.md (numpy aliases, distutils.version, Ada sm_89 TorchScript).
export PYTHONPATH="$ROOT/compat:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH_FOLLOWER_ROOT="$ROOT"
