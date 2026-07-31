"""Lightweight INT8 GEMM + dequant helper.

Deliberately does not import NVRTC/tilelang, so the ``eager`` backend remains a
true no-JIT compatibility path. ``comfy_kitchen`` is an optional dependency: if
present, its ``_int8_matmul_accumulate`` (cuBLASLt IMMA with Turing padding) is
preferred; otherwise ``torch._int_mm`` is used.
"""
from __future__ import annotations

import torch


def _int8_matmul_accumulate():
    """Return the best available INT8 matmul primitive."""
    try:
        from comfy_kitchen.backends.eager.quantization import _int8_matmul_accumulate as fn
        return fn
    except ImportError:
        return torch._int_mm


def int8_matmul_dequant_chunked(
    xq: torch.Tensor,
    xscale: torch.Tensor,
    wq: torch.Tensor,
    wscale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    max_temp_bytes: int = 256 * 1024 * 1024,
) -> torch.Tensor:
    """INT8 GEMM + dequant with a bounded int32/float32 temporary footprint.

    Computes ``acc = xq @ wq.T`` (int32), then
    ``out = acc.float() * xscale[:,None] * wscale[None,:] (+ bias)``.

    Args:
        xq: [M, K] int8 activation.
        xscale: [M] float32 per-row activation scale.
        wq: [N, K] int8 weight.
        wscale: [N] or scalar float32 weight scale.
        bias: optional [N] bias.
        out_dtype: output dtype (typically fp16/bf16).
        max_temp_bytes: cap on the int32/float32 temp per chunk.
    """
    m = xq.shape[0]
    n = wq.shape[0]
    rows_per_chunk = max(1, min(m, max_temp_bytes // max(1, n * 8)))
    weight_t = wq.t()  # transposed view; do not copy the full weight per call.
    weight_scale = wscale.to(device=xq.device, dtype=torch.float32).reshape(1, -1)
    if weight_scale.numel() not in (1, n):
        raise ValueError(
            f"weight scale must be scalar or length {n}, got {tuple(wscale.shape)}"
        )

    mm = _int8_matmul_accumulate()
    output = torch.empty((m, n), device=xq.device, dtype=out_dtype)
    bias_float = None if bias is None else bias.to(
        device=xq.device, dtype=torch.float32
    ).reshape(1, -1)
    for start in range(0, m, rows_per_chunk):
        end = min(start + rows_per_chunk, m)
        acc = mm(xq[start:end], weight_t)
        part = acc.float()
        part.mul_(xscale[start:end].float().reshape(-1, 1))
        part.mul_(weight_scale)
        if bias_float is not None:
            part.add_(bias_float)
        output[start:end].copy_(part)
        del acc, part
    return output
