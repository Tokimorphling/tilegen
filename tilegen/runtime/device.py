"""Runtime GPU capability probe for portable kernel tuning.

Kernel launch parameters (notably dynamic shared-memory budgets) differ a lot
across architectures::

    arch        per-block opt-in shared
    Turing 7.5   64 KB
    Ampere 8.0  164 KB   (A100)
    Ampere 8.6  100 KB   (RTX 30xx)
    Ada    8.9  100 KB   (RTX 40xx)
    Hopper 9.0  227 KB

This module queries the actual limits via the CUDA driver API (cuda-python) so
kernels tune per device with no hard-coded constants, falling back to
conservative Turing-safe values if the driver query fails.
"""
from __future__ import annotations

import functools


# Conservative fallback if the driver query is unavailable (matches Turing).
_FALLBACK = {
    "sm": (7, 5),
    "shared_per_block": 49152,
    "shared_per_block_optin": 65536,
    "shared_per_sm": 65536,
    "sm_count": 1,
    "name": "unknown",
}


@functools.lru_cache(maxsize=4)
def probe(device_index: int = 0) -> dict:
    """Return a dict of the GPU's shared-memory / SM limits. Cached per device."""
    try:
        import cuda.bindings.driver as cd

        cd.cuInit(0)
        dev = cd.cuDeviceGet(device_index)[1]

        def attr(a):
            return cd.cuDeviceGetAttribute(a, dev)[1]

        A = cd.CUdevice_attribute
        major = attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = attr(A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        return {
            "sm": (int(major), int(minor)),
            "shared_per_block": int(attr(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)),
            "shared_per_block_optin": int(attr(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)),
            "shared_per_sm": int(attr(A.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)),
            "sm_count": int(attr(A.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)),
            "name": "cuda:%d" % device_index,
        }
    except Exception:
        return dict(_FALLBACK)


def usable_shared_bytes(device_index: int = 0, headroom: int = 1024) -> int:
    """Max dynamic shared memory requestable per block, minus a small headroom.

    Uses the opt-in ceiling (what ``cuFuncSetAttribute`` can raise a kernel to).
    """
    caps = probe(device_index)
    return max(0, caps["shared_per_block_optin"] - headroom)


def pick_block_m_rowbuf(
    K: int,
    itemsize_accum: int = 4,
    device_index: int = 0,
    cap: int = 64,
) -> int:
    """block_M for a kernel that stages a [block_M, K] accum row in shared.

    Shared usage scales with K: ``block_M * K * itemsize``. Used by the v1 path.
    """
    budget = usable_shared_bytes(device_index)
    bm = max(1, budget // (K * itemsize_accum))
    return min(bm, cap)


def pick_block_m_grouped(
    group_size: int = 256,
    in_itemsize: int = 2,
    accum_itemsize: int = 4,
    device_index: int = 0,
    cap: int = 64,
) -> int:
    """block_M for the v2 grouped kernel (shared independent of K).

    Shared usage is ``block_M * group_size * (in_itemsize + accum_itemsize)``.
    """
    budget = usable_shared_bytes(device_index)
    per_row = group_size * (in_itemsize + accum_itemsize)
    bm = max(8, budget // per_row)
    bm = (bm // 8) * 8  # round down to a multiple of 8 (warp-friendly)
    return max(8, min(bm, cap))


def hadamard_fits_shared(group_size: int = 256, in_itemsize: int = 2, device_index: int = 0) -> bool:
    """Whether the full ``group_size x group_size`` Hadamard fits in shared memory.

    On Turing (64KB) a 256x256 fp16 H is 128KB and does not fit, so the caller
    must use a global-H rotation path.
    """
    h_bytes = group_size * group_size * in_itemsize
    return h_bytes + (group_size * 64 * in_itemsize) <= usable_shared_bytes(device_index)


if __name__ == "__main__":
    import json

    caps = probe()
    print(json.dumps(caps, indent=2))
    print("usable_shared:", usable_shared_bytes(), "bytes")
    print("v2 block_M (grouped):", pick_block_m_grouped())
    print("Hadamard 256 fits shared:", hadamard_fits_shared())
