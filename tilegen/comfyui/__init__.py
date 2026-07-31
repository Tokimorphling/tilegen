"""Native ComfyUI integration for tilegen.

Install ``tilegen[comfyui]``, then copy or link the repository's top-level
``comfyui`` loader into ``ComfyUI/custom_nodes/tilegen``. The INT8 ConvRot
backend is registered automatically on startup:

* Registers a selective ``tilegen`` backend ahead of the accelerated backends.
  Eligible wide ConvRot calls use FHT; every unsupported or disabled call falls
  through to ComfyUI's normal CUDA/Triton/eager priority.
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
        TileGenConvRotModelConfig,
    )

    NODE_CLASS_MAPPINGS = {
        "ConvRotTestTensors": ConvRotTestTensors,
        "QuantizeConvRotWeight": QuantizeConvRotWeight,
        "Int8ConvRotLinear": Int8ConvRotLinear,
        "TileGenConvRotModelConfig": TileGenConvRotModelConfig,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ConvRotTestTensors": "ConvRot Test Tensors",
        "QuantizeConvRotWeight": "Quantize ConvRot Weight (INT8)",
        "Int8ConvRotLinear": "INT8 ConvRot Linear (tilegen)",
        "TileGenConvRotModelConfig": "TileGen ConvRot Runtime (MODEL passthrough)",
    }
except Exception:  # pragma: no cover - import errors should not break ComfyUI
    LOGGER.exception("tilegen: nodes failed to import")

# The backend is registered only when comfy_kitchen is present (i.e. inside a
# ComfyUI runtime) and not explicitly disabled.
_BACKEND_INSTALLED = False
try:
    from . import backend as _backend

    if os.environ.get("TILEGEN_CONVROT_BACKEND", "auto").strip().lower() != "off":
        _BACKEND_INSTALLED = _backend.install()
except Exception:  # pragma: no cover
    LOGGER.exception("tilegen: backend registration failed")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
