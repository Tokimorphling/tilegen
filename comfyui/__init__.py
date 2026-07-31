"""tilegen — ComfyUI custom node package.

Drop this directory (or a symlink to it) into ``ComfyUI/custom_nodes/`` and the
INT8 ConvRot acceleration backend is registered automatically on startup:

* Registers a ``tilegen`` backend into ``comfy_kitchen``'s registry with a
  priority between the native accelerated backends and ``eager``. Any model
  that calls ``int8_linear(convrot=True)`` (e.g. SCAIL-2 int8_convrot) then
  dispatches through the tilegen FHT kernel for large K, and a stable PyTorch
  path for small K.
* Exposes standalone ComfyUI nodes (``QuantizeConvRotWeight``,
  ``Int8ConvRotLinear``, ``ConvRotTestTensors``) for manual / test wiring.

Backend selection is controlled by the ``TILEGEN_CONVROT_BACKEND`` env var:

* ``auto`` (default): FHT for wide K (faster), eager torch for smaller K.
* ``fht``: force FHT for every supported K (mainly for benchmarking).
* ``eager``: only the stable PyTorch implementation.
* ``off``: do not register the backend at all.

A minimum-K threshold for the FHT path defaults to 5120 on Turing (sm75) and
8192 elsewhere; override with ``TILEGEN_FHT_MIN_K`` (0 = always use FHT).
"""
from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger("tilegen")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

# Nodes are always safe to import (they only use the core tilegen kernels).
try:
    from .nodes import (
        ConvRotTestTensors,
        Int8ConvRotLinear,
        QuantizeConvRotWeight,
    )

    NODE_CLASS_MAPPINGS = {
        "ConvRotTestTensors": ConvRotTestTensors,
        "QuantizeConvRotWeight": QuantizeConvRotWeight,
        "Int8ConvRotLinear": Int8ConvRotLinear,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ConvRotTestTensors": "ConvRot Test Tensors",
        "QuantizeConvRotWeight": "Quantize ConvRot Weight (INT8)",
        "Int8ConvRotLinear": "INT8 ConvRot Linear (tilegen)",
    }
except Exception:  # pragma: no cover - import errors should not break ComfyUI
    LOGGER.exception("tilegen: nodes failed to import")

# The backend is registered only when comfy_kitchen is present (i.e. inside a
# ComfyUI runtime) and not explicitly disabled.
_BACKEND_INSTALLED = False
if os.environ.get("TILEGEN_CONVROT_BACKEND", "auto").strip().lower() != "off":
    try:
        from .backend import install as _install_backend

        _BACKEND_INSTALLED = _install_backend()
    except Exception:  # pragma: no cover
        LOGGER.exception("tilegen: backend registration failed")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
