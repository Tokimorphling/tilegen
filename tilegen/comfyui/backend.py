"""Register tilegen as a selective comfy_kitchen ConvRot backend.

When ComfyUI loads this custom node, :func:`install` registers a backend named
``tilegen`` into comfy_kitchen's :class:`~comfy_kitchen.registry.BackendRegistry`.
It is checked before the native accelerated backends, but its call rule accepts
only shapes that can use FHT. Models that call ``int8_linear(convrot=True)``
(SCAIL-2 int8_convrot and friends) then dispatch as follows:

* wide K -> the fused NVRTC FHT activation kernel (tilegen.kernels)
* small K / unsupported -> the existing ComfyUI CUDA/Triton/eager priority

The INT8 GEMM + dequant reuses comfy_kitchen's IMMA path when available.

``comfy_kitchen`` is a hard requirement for backend registration (the nodes in
:mod:`tilegen.comfyui.nodes` work without it). When comfy_kitchen is absent,
:func:`install` is a no-op returning False.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import Mapping

import torch

from ..quant.int8_ops import int8_matmul_dequant_chunked
from ..runtime.device import usable_shared_bytes

LOGGER = logging.getLogger("tilegen.backend")

BACKEND_NAME = "tilegen"
_INSTALLED = False
_FAILED_FHT_SHAPES: set[tuple] = set()
_LOGGED_DECISIONS: set[tuple] = set()


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


def _diagnostics_enabled() -> bool:
    return os.environ.get("TILEGEN_DIAGNOSTICS", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


def _log_decision(decision: str, reason: str, x, weight, group_size: int) -> None:
    if not _diagnostics_enabled() or not isinstance(x, torch.Tensor) or x.ndim < 2:
        return
    k = x.shape[-1]
    m = x.numel() // max(1, k)
    n = weight.shape[0] if isinstance(weight, torch.Tensor) and weight.ndim == 2 else "?"
    key = (decision, reason, m, k, n, x.dtype, group_size)
    if key in _LOGGED_DECISIONS:
        return
    _LOGGED_DECISIONS.add(key)
    LOGGER.warning(
        "tilegen shape %s: M=%s K=%s N=%s dtype=%s group=%s reason=%s",
        decision, m, k, n, x.dtype, group_size, reason,
    )


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


def _fht_applicable(kwargs: Mapping[str, object]):
    """Registry rule that lets unsupported shapes fall through to CUDA."""
    from comfy_kitchen.constraints import ValidationResult

    mode = _mode()
    x = kwargs.get("x")
    weight = kwargs.get("weight")
    convrot = kwargs.get("convrot", False)
    group_size = int(kwargs.get("convrot_groupsize", 256))
    if not convrot:
        return ValidationResult.fail("convrot", "TileGen only handles ConvRot")
    if mode not in {"auto", "fht"}:
        _log_decision("KEEP_CUDA", f"mode={mode}", x, weight, group_size)
        return ValidationResult.fail("__tilegen__", f"mode={mode}")
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim < 2:
        return ValidationResult.fail("x", "CUDA tensor with at least 2 dims required")
    if x.dtype not in (torch.float16, torch.bfloat16):
        return ValidationResult.fail("x", "fp16/bf16 input required")
    k = x.shape[-1]
    if group_size != 256 or k % 256:
        _log_decision("KEEP_CUDA", "unsupported group/K", x, weight, group_size)
        return ValidationResult.fail("convrot_groupsize", "requires group=256 and K divisible by 256")
    min_k = 0 if mode == "fht" else _fht_min_k(x.device.index or 0)
    if k < min_k:
        _log_decision("KEEP_CUDA", f"K below min_k={min_k}", x, weight, group_size)
        return ValidationResult.fail("x", "K below TileGen threshold")
    if k * 4 > usable_shared_bytes(x.device.index or 0, headroom=2048):
        _log_decision("KEEP_CUDA", "FHT row exceeds shared memory", x, weight, group_size)
        return ValidationResult.fail("x", "FHT row exceeds shared memory")
    shape_key = (x.numel() // k, k, x.dtype, x.device.index)
    if shape_key in _FAILED_FHT_SHAPES:
        _log_decision("KEEP_CUDA", "FHT previously failed", x, weight, group_size)
        return ValidationResult.fail("x", "FHT previously failed for this shape")
    _log_decision("ACCEPT_FHT", "all constraints passed", x, weight, group_size)
    return ValidationResult.ok()


def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
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
        from comfy_kitchen.backends.eager.quantization import (
            int8_linear as eager_int8_linear,
        )

        return eager_int8_linear(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            input_act=input_act,
        )
    if input_act not in (None, "none"):
        from comfy_kitchen.backends._activations import apply_input_act

        x = apply_input_act(x, input_act)
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
    """Register selective FHT dispatch ahead of the normal backends."""
    global _INSTALLED
    mode = _mode()
    if mode == "off" or _INSTALLED:
        return _INSTALLED

    try:
        from comfy_kitchen.registry import registry

        device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        fht_min_k = 0 if mode == "fht" else _fht_min_k(device_index)
        from comfy_kitchen.constraints import FunctionConstraints

        base = registry.get_constraints("eager", "int8_linear")
        if base is None:
            raise RuntimeError("comfy_kitchen eager int8_linear constraints are unavailable")
        constraints = FunctionConstraints(
            params=base.params,
            default_devices=base.default_devices,
            min_compute_capability=base.min_compute_capability,
            call_rules=(*getattr(base, "call_rules", ()), _fht_applicable),
        )
        registry.register(BACKEND_NAME, sys.modules[__name__], {"int8_linear": constraints})
        existing = [name for name in registry._priority if name != BACKEND_NAME]
        registry.set_priority([BACKEND_NAME, *existing])
        _INSTALLED = True
        LOGGER.warning(
            "tilegen selective FHT dispatch enabled: mode=%s FHT min K=%s temp=%s MiB priority=%s",
            mode,
            fht_min_k,
            os.environ.get("TILEGEN_INT8_TEMP_MB", "256"),
            registry._priority,
        )
        return True
    except Exception:
        LOGGER.exception("tilegen: could not register the ConvRot backend")
        return False
