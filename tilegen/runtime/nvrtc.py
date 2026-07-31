"""Platform-independent NVRTC bootstrap and version probe.

How NVRTC is obtained
---------------------
NVRTC is **not installed separately by the user**. It ships inside whichever
torch wheel they pip-installed (``torch/lib/nvrtc64_*.dll`` on Windows,
``torch/lib/libnvrtc.so.*`` on Linux). This module biases the dynamic loader
toward that bundled copy so no system CUDA toolkit is required.

Fallback chain, in order:

1. torch's bundled NVRTC (preferred — matches torch's own CUDA runtime).
2. The ``nvidia-cuda-nvrtc-cuXX`` pip package (``nvidia/cuda_nvrtc/``), if
   importable. On Windows the DLL sits in ``bin/``, on Linux the .so in ``lib/``.
3. Whatever ``cuda.bindings.nvrtc`` resolves on its own (system CUDA toolkit on
   ``PATH`` / ``CUDA_HOME`` / ``LD_LIBRARY_PATH``).

Platform notes
--------------
* **Windows**: preload via ``os.add_dll_directory`` + ``ctypes.CDLL``. This is
  validated (torch 2.6.0+cu124 / NVRTC 12.4).
* **Linux**: torch's manylinux wheel also bundles ``libnvrtc.so`` in
  ``torch/lib``; preload with ``ctypes.CDLL(..., RTLD_GLOBAL)`` so the soname is
  registered and ``cuda-python`` reuses it instead of re-resolving. If torch's
  copy is absent, the ``nvidia-cuda-nvrtc-cuXX`` dependency provides one.

CUDA 13 / Blackwell adaptation
------------------------------
Selection is fully version-agnostic: the bundled DLL/.so is located by glob
(``nvrtc64_*.dll`` / ``libnvrtc.so*``), so a torch built with cu13x (which
bundles NVRTC 13) is picked up automatically. The kernel's target architecture
(``--gpu-architecture=sm_{major}{minor}``) is queried at runtime from
``torch.cuda.get_device_capability()``, so sm100/sm120 on Blackwell are handled
by NVRTC 13 without any code change.

The only hard requirement is that the resolved NVRTC be new enough to emit code
for the current GPU's arch:

* the tilelang template path needs NVRTC >= 12.0 (c++20 templates);
* the native CUDA-C kernel path needs only NVRTC >= 11.0 (plain CUDA C), but it
  still cannot target an arch newer than itself.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

_BOOTSTRAPPED = False
_RESOLVED_VERSION: tuple[int, int] | None = None


def _torch_lib_dir() -> str | None:
    try:
        import torch

        return os.path.join(os.path.dirname(torch.__file__), "lib")
    except ImportError:
        return None


def _glob_nvrtc(lib_dir: str) -> str | None:
    """Find an nvrtc shared library in ``lib_dir`` (version-agnostic).

    Windows: ``nvrtc64_*.dll``. Linux: ``libnvrtc.so*``. Picks the
    lexicographically-last match (highest version when several coexist).
    """
    if not lib_dir or not os.path.isdir(lib_dir):
        return None
    if sys.platform == "win32":
        matches = sorted(glob.glob(os.path.join(lib_dir, "nvrtc64_*.dll")))
    else:
        matches = sorted(glob.glob(os.path.join(lib_dir, "libnvrtc.so*")))
    return matches[-1] if matches else None


def _find_torch_nvrtc() -> str | None:
    """Locate the NVRTC shared library bundled with torch, or None.

    Both the Windows and Linux (manylinux) torch wheels bundle NVRTC in
    ``torch/lib``.
    """
    return _glob_nvrtc(_torch_lib_dir())


def _find_pip_nvrtc() -> str | None:
    """Locate the nvidia-cuda-nvrtc pip package's lib, if installed.

    The package installs under ``nvidia/cuda_nvrtc/``; the shared library lives
    in ``bin/`` on Windows and ``lib/`` on Linux.
    """
    try:
        import nvidia  # type: ignore  # nvidia-cuda-nvrtc-cuXX

        base = os.path.dirname(nvidia.__file__)
        sub = os.path.join(base, "cuda_nvrtc")
        for leaf in ("bin", "lib"):
            found = _glob_nvrtc(os.path.join(sub, leaf))
            if found:
                return found
    except Exception:
        return None
    return None


def _nvrtc_version() -> tuple[int, int] | None:
    """Best-effort query of the NVRTC version cuda-python currently binds.

    ``cuda.bindings.nvrtc.nvrtcVersion()`` returns ``(nvrtcResult, major, minor)``.
    """
    try:
        from cuda.bindings import nvrtc

        err, major, minor = nvrtc.nvrtcVersion()
        if int(err) != 0:
            return None
        return (int(major), int(minor))
    except Exception:
        return None


def _preload(dll: str) -> None:
    lib_dir = os.path.dirname(dll)
    try:
        if sys.platform == "win32":
            os.add_dll_directory(lib_dir)
            ctypes.CDLL(dll)
        else:
            try:
                os.add_dll_directory(lib_dir)
            except (AttributeError, OSError):
                pass
            # RTLD_GLOBAL registers the soname so cuda-python's dlopen reuses
            # this copy instead of re-resolving from LD_LIBRARY_PATH.
            ctypes.CDLL(dll, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # cuda-python will resolve whatever it can from the system.


def ensure_nvrtc() -> tuple[int, int] | None:
    """Preload the best available NVRTC and return its version (or None).

    Idempotent and thread-safe enough for import-time use. After this returns,
    ``cuda.bindings.nvrtc`` resolves to the preloaded NVRTC.
    """
    global _BOOTSTRAPPED, _RESOLVED_VERSION
    if _BOOTSTRAPPED:
        return _RESOLVED_VERSION

    for finder in (_find_torch_nvrtc, _find_pip_nvrtc):
        dll = finder()
        if dll:
            _preload(dll)
            break

    _RESOLVED_VERSION = _nvrtc_version()
    _BOOTSTRAPPED = True
    return _RESOLVED_VERSION


def require_nvrtc(min_version: tuple[int, int] = (11, 0)) -> tuple[int, int]:
    """Preload NVRTC and raise if it is older than ``min_version``.

    Use this from a compile path that genuinely needs a floor version. The
    error message distinguishes "no NVRTC found" from "too old".
    """
    version = ensure_nvrtc()
    if version is None:
        raise RuntimeError(
            "No NVRTC found. Install torch (it bundles NVRTC) or the "
            "nvidia-cuda-nvrtc-cuXX pip package, or set CUDA_HOME to a CUDA "
            "toolkit install."
        )
    if version < min_version:
        raise RuntimeError(
            f"NVRTC {version[0]}.{version[1]} is too old; need >= "
            f"{min_version[0]}.{min_version[1]}. Upgrade torch (it bundles a "
            f"matching NVRTC) or install a newer CUDA toolkit."
        )
    return version


# Back-compat aliases -------------------------------------------------------
def ensure_nvrtc_12() -> tuple[int, int] | None:
    """Preload NVRTC and return its version; kept for back-compat.

    Does NOT enforce >= 12 — use :func:`require_nvrtc` for that. Renamed to
    :func:`ensure_nvrtc` because selection is now version-agnostic.
    """
    return ensure_nvrtc()


def bootstrap() -> None:
    ensure_nvrtc()
