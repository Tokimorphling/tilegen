"""Common Hadamard helpers shared by ConvRot kernels and weight quantization.

The ConvRot rotation uses the radon-normalized Sylvester-Hadamard matrix of
order ``group_size`` (a power of 4), built by repeated Kronecker product of the
4x4 ``H4`` core. ``group_size=256`` is the specialization the FHT kernel uses
(four radix-4 butterfly stages in shared memory).
"""
from __future__ import annotations

import functools
import math

import torch

_H4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]


def _validate_group_size(group_size: int) -> None:
    if group_size < 4 or (group_size & (group_size - 1)) != 0:
        raise ValueError(f"group_size must be a power of two >= 4, got {group_size}")
    if math.log(group_size, 4) % 1 != 0:
        raise ValueError(f"group_size must be a power of 4, got {group_size}")


@functools.lru_cache(maxsize=16)
def _hadamard(device_type: str, device_index: int, dtype: torch.dtype, size: int) -> torch.Tensor:
    """Cached, radon-normalized Hadamard matrix of order ``size`` (a power of 4)."""
    _validate_group_size(size)
    device = torch.device(device_type, device_index) if device_type == "cuda" else torch.device(device_type)
    h4 = torch.tensor(_H4, dtype=dtype, device=device)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    return h / (size ** 0.5)


def build_hadamard(size: int, device, dtype: torch.dtype) -> torch.Tensor:
    """Return the radon-normalized Hadamard matrix of order ``size``.

    Args:
        size: power of 4 (e.g. 4, 16, 64, 256).
        device: torch device.
        dtype: torch dtype.
    """
    if isinstance(device, torch.device):
        device_type = device.type
        device_index = device.index or 0
    else:
        device_type = str(device)
        device_index = 0
    return _hadamard(device_type, device_index, dtype, size)


def rotate_by_group(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate ``x [*, K]`` by group Hadamard: reshape to groups and matmul ``h``."""
    leading = x.shape[:-1]
    k = x.shape[-1]
    if k % group_size:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    n_groups = k // group_size
    xg = x.reshape(*leading, n_groups, group_size)
    rotated = torch.matmul(xg, h.to(x.dtype))
    return rotated.reshape(*leading, k)
