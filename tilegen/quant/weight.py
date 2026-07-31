"""Offline weight-side ConvRot quantization: group Hadamard rotation + per-channel int8.

Done once per weight and cached. Mirrors comfy_kitchen's eager
``int8_linear(convrot=True)`` numerics exactly, so a quantized weight produced
here is interoperable with any backend that consumes the same layout.
"""
from __future__ import annotations

import torch

from ..kernels.hadamard import build_hadamard, rotate_by_group


def quantize_convrot_weight(
    weight: torch.Tensor,
    group_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate by group Hadamard + per-channel (per-row) int8 quantization.

    Args:
        weight: [N, K] fp16/bf16 weight matrix.
        group_size: rotation group size (power of 4).

    Returns:
        (wq [N, K] int8, wscale [N] float32).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [N,K], got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")

    h = build_hadamard(group_size, weight.device, weight.dtype)
    wrot = rotate_by_group(weight, h, group_size)

    wabs = wrot.abs().amax(dim=-1, keepdim=True)
    wscale = (wabs.float() / 127.0).clamp(min=1e-30)
    wq = torch.round(wrot / wscale.to(wrot.dtype)).clamp(-128, 127).to(torch.int8)
    return wq, wscale.reshape(n)
