"""Compatibility shims for running this IsaacGym stack on the local machine.

CPython imports ``sitecustomize`` automatically at interpreter start-up when its
directory is on ``PYTHONPATH``, so these run before IsaacGym and torch are
imported.  ``run_local.sh`` puts this directory first on ``PYTHONPATH``.  No
installed package or virtualenv is modified.

Two environmental assumptions of the original workstation no longer hold:

1. NumPy >= 1.24 removed the ``np.float`` and ``np.int`` aliases.
   IsaacGym's ``isaacgym/torch_utils.py`` evaluates ``dtype=np.float`` as a
   default argument at import time, so importing IsaacGym aborts with
   ``AttributeError: module 'numpy' has no attribute 'float'``.
   Restore the removed aliases; a NumPy that still provides them is unaffected.

2. torch 1.10's tensorboard shim does ``import distutils`` followed by
   ``distutils.version.LooseVersion``, relying on something else having
   imported the ``distutils.version`` submodule first.  Nothing does that with
   setuptools >= 60, so the attribute lookup fails.  Import the submodule here.

3. IsaacGym's TorchScript helpers cannot be compiled by the CUDA 11.3 nvrtc
   bundled with torch 1.10 on an RTX 40-series (sm_89) GPU.  See
   ``isaacgym_ada_compat`` for the import hook that swaps them for eager
   equivalents.
"""

import warnings

import distutils.version  # noqa: F401  imported for the side effect on distutils

import numpy as np

from isaacgym_ada_compat import install as _install_ada_compat

with warnings.catch_warnings():
    # 'bool'/'object'/'str' probes emit FutureWarning on NumPy < 2;
    # importing distutils.version emits a DeprecationWarning on Python 3.8+.
    warnings.simplefilter("ignore")

    # Aliases removed in NumPy 1.24, mapped to their documented replacements.
    for _name, _replacement in {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
        "str": str,
        "complex": complex,
    }.items():
        if not hasattr(np, _name):
            setattr(np, _name, _replacement)

_install_ada_compat()
