"""Register tilegen as a comfy_kitchen ConvRot backend.

When ComfyUI loads this custom node, :func:`install` registers a backend named
``tilegen`` into comfy_kitchen's :class:`~comfy_kitchen.registry.BackendRegistry`,
positioned between the native accelerated backends and ``eager``. Models that
call ``int8_linear(convrot=True)`` (SCAIL-2 int8_convrot and friends) then
dispatch here:

* wide K -> the fused NVRTC FHT activation kernel (tilegen.kernels)
* small K / unsupported -> the stable PyTorch Hadamard rotation

The INT8 GEMM + dequant reuses comfy_kitchen's IMMA path when available, so the
only thing tilegen replaces is the (slow, on Turing) activation rotation+
quantization — exactly where the upstream int8_convrot path regresses.

``comfy_kitchen`` is a hard requirement for backend registration (the nodes in
:mod:`tilegen.comfyui.nodes` work without it). When comfy_kitchen is absent,
:func:`install` is a no-op returning False.
"""
from __future__ import annotations

import functools
import logging
import os
import sys

import torch

from ..runtime.device import usable_shared_bytes
from ..quant.int8_ops import int8_matmul_dequant_chunked

LOGGER = logging.getLogger("tilegen.backend")

BACKEND_NAME = "tilegen"
_INSTALLED = False
_FAILED_FHT_SHAPES: set[tuple] = set()


def _mode() -> str:
    value = os.environ.get("TILEGEN_CONVROT_BACKEND", "auto").strip().lower()
    if value not in {"off", "auto", "fht", "eager"}:
        LOGGER.warning("Unknown TILEGEN_CONVROT_BACKEND=%r; using auto", value)
        return "auto"
    return value


def _default_fht_min_k(device_index: int = 0) -> int:
    """Conservative auto threshold for the FHT path, per device.

    The native NVRTC kernel is the validated path; the original tilelang
    lowering overran shared memory under occupancy on some stacks (see
    tilegen.kernels._tilelang_fht). Turing's measured crossover is K=5120.
    """
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(device_index) == (7, 5):
            return 5120
    except (AssertionError, RuntimeError):
        pass
    return 8192


def _fht_min_k(device_index: int = 0) -> int:
    """Env override or the per-device auto threshold."""
    value = os.environ.get("TILEGEN_FHT_MIN_K")
    if value is None:
        return _default_fht_min_k(device_index)
    try:
        return max(0, int(value))
    except ValueError:
        default = _default_fht_min_k(device_index)
        LOGGER.warning("Invalid TILEGEN_FHT_MIN_K=%r; using %d", value, default)
        return default


@functools.lru_cache(maxsize=16)
def _hadamard(device_type: str, device_index: int, dtype: torch.dtype, size: int):
    from ..kernels.hadamard import build_hadamard

    device = torch.device(device_type, device_index) if device_type == "cuda" else torch.device(device_type)
    return build_hadamard(size, device, dtype)


def _eager_quantize_convrot(x: torch.Tensor, group_size: int):
    """Pure-torch activation ConvRot quant (stable reference / fallback)."""
    m, k = x.shape
    h = _hadamard(x.device.type, x.device.index or 0, x.dtype, group_size)
    rotated = torch.matmul(x.reshape(m, k // group_size, group_size), h).reshape(m, k)
    scale = (rotated.abs().amax(dim=-1).float() / 127.0).clamp_min(1e-30)
    qdata = torch.round(rotated / scale[:, None].to(rotated.dtype)).clamp(-128, 127).to(torch.int8)
    return qdata, scale


def _can_use_fht(x: torch.Tensor, group_size: int, mode: str) -> bool:
    if mode not in {"auto", "fht"}:
        return False
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if group_size != 256 or x.shape[1] % 256:
        return False
    min_k = 0 if mode == "fht" else _fht_min_k(x.device.index or 0)
    if x.shape[1] < min_k:
        return False
    # The kernel holds one float32 row in dynamic shared memory.
    return x.shape[1] * 4 <= usable_shared_bytes(x.device.index or 0, headroom=2048)


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """Backend-compatible replacement for comfy_kitchen.int8_linear."""
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"input and weight inner dimensions differ: {x.shape[-1]} != {weight.shape[-1]}"
        )
    out_dtype = out_dtype or x.dtype
    mode = _mode()
    if not convrot or mode == "eager":
        # This backend exists specifically for ConvRot. Preserve comfy_kitchen's
        # behavior for every other INT8 layout. Explicit eager mode is also a
        # true no-JIT compatibility path and therefore delegates directly.
        from comfy_kitchen.backends.eager.quantization import int8_linear as eager_int8_linear

        return eager_int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
        )
    original_shape = tuple(x.shape)
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight = weight.to(device=x.device, dtype=torch.int8)

    if convrot:
        if x2d.shape[1] % convrot_groupsize:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide K={x2d.shape[1]}"
            )
        shape_key = (x2d.shape[0], x2d.shape[1], x2d.dtype, x2d.device.index)
        if _can_use_fht(x2d, convrot_groupsize, mode) and shape_key not in _FAILED_FHT_SHAPES:
            try:
                from ..kernels.int8_convrot import quantize_convrot_fht

                xq, xscale = quantize_convrot_fht(x2d, convrot_groupsize)
            except Exception:
                _FAILED_FHT_SHAPES.add(shape_key)
                LOGGER.exception(
                    "FHT failed for M=%d K=%d; using eager ConvRot",
                    x2d.shape[0],
                    x2d.shape[1],
                )
                xq, xscale = _eager_quantize_convrot(x2d, convrot_groupsize)
        else:
            xq, xscale = _eager_quantize_convrot(x2d, convrot_groupsize)
    else:
        xscale = (x2d.abs().amax(dim=-1).float() / 127.0).clamp_min(1e-30)
        xq = torch.round(x2d / xscale[:, None].to(x2d.dtype)).clamp(-128, 127).to(torch.int8)

    max_temp_mb = max(16, int(os.environ.get("TILEGEN_INT8_TEMP_MB", "256")))
    out = int8_matmul_dequant_chunked(
        xq,
        xscale,
        weight,
        weight_scale,
        bias,
        out_dtype,
        max_temp_bytes=max_temp_mb * 1024 * 1024,
    )
    return out.reshape(*original_shape[:-1], weight.shape[0])


def install() -> bool:
    """Register after native accelerated backends and before eager. Idempotent."""
    global _INSTALLED
    mode = _mode()
    if mode == "off" or _INSTALLED:
        return _INSTALLED

    try:
        from comfy_kitchen.registry import registry

        device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        fht_min_k = 0 if mode == "fht" else _fht_min_k(device_index)
        constraints = registry.get_constraints("eager", "int8_linear")
        if constraints is None:
            raise RuntimeError("comfy_kitchen eager int8_linear constraints are unavailable")
        registry.register(BACKEND_NAME, sys.modules[__name__], {"int8_linear": constraints})
        existing = [name for name in registry._priority if name != BACKEND_NAME]
        accelerated = [name for name in existing if name != "eager"]
        priority = accelerated + [BACKEND_NAME]
        if "eager" in existing:
            priority.append("eager")
        registry.set_priority(priority)
        _INSTALLED = True
        LOGGER.warning(
            "tilegen ConvRot backend enabled: mode=%s  FHT min K=%s  temp=%s MiB",
            mode,
            fht_min_k,
            os.environ.get("TILEGEN_INT8_TEMP_MB", "256"),
        )
        return True
    except Exception:
        LOGGER.exception("tilegen: could not register the ConvRot backend")
        return False
