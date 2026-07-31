"""tilegen: tilelang / NVRTC acceleration kernels for DiT inference.

The package is platform-independent. Kernel compilation uses NVRTC, which is
bundled with torch (cu12x); :mod:`tilegen.runtime.nvrtc` biases the loader
toward that copy so no system CUDA toolkit is required.
"""
from __future__ import annotations

__version__ = "0.1.0"
