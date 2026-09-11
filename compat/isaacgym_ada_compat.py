"""Make IsaacGym Preview 4's torch helpers run on Ada (sm_89) GPUs.

``isaacgym/torch_utils.py`` decorates its helpers with ``@torch.jit.script``.
Calling a ``ScriptFunction`` on a CUDA tensor makes torch 1.10 compile a fused
kernel through nvrtc, and the CUDA 11.3 nvrtc that ships with ``torch==1.10.0+cu113``
has no ``sm_89`` target.  On an RTX 40-series card that aborts with::

    RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)

The helpers are plain elementwise/tensor expressions, so their eager Python
definitions are numerically identical and simply skip nvrtc.  This module
installs an import hook that swaps the ``ScriptFunction`` attributes of
``isaacgym.torch_utils`` for the eager functions as soon as that module is
imported.  The installed IsaacGym package is not modified; nothing is patched
on non-Ada machines beyond losing TorchScript fusion in these helpers.

Install with::

    from isaacgym_ada_compat import install
    install()
"""

import importlib.abc
import sys
from pathlib import Path

_TARGET = "isaacgym.torch_utils"


def _eager_definitions(module):
    """Re-execute the module source with ``torch.jit.script`` as a no-op.

    Returns a namespace mapping helper name -> plain Python function, so the
    eager bodies always match the installed source rather than being retyped
    here.
    """
    import torch

    source = Path(module.__file__).read_text(encoding="utf-8")
    namespace = {}
    original_script = torch.jit.script
    # @torch.jit.script must work both bare and with keyword arguments.
    torch.jit.script = lambda obj=None, **kwargs: obj if obj is not None else (lambda f: f)
    try:
        exec(compile(source, module.__file__, "exec"), namespace)
    finally:
        torch.jit.script = original_script
    return namespace


def _swap_scripted_helpers(module):
    import torch

    namespace = _eager_definitions(module)
    for name, value in list(vars(module).items()):
        if isinstance(value, torch.jit.ScriptFunction) and name in namespace:
            setattr(module, name, namespace[name])


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, post_exec):
        self._wrapped = wrapped
        self._post_exec = post_exec

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        self._post_exec(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, _swap_scripted_helpers)
                return spec
        return None


def install():
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
