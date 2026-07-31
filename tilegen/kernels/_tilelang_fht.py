"""tilelang lowering of the fused ConvRot activation quantizer (radix-4 FHT).

This is the framework's intended implementation of the same op as the native
NVRTC kernel in :mod:`tilegen.kernels.int8_convrot`. It expresses the four
radix-4 butterfly stages of the H256 Kronecker Hadamard in tilelang, keeping
the full rotated row in shared memory so the row-wise absmax (and thus the
per-row int8 scale) is exact with no second pass.

Note
----
On some toolchains (tilelang 0.1.12 / NVRTC 12.4 / sm75) ptxas software-
pipelines the vectorized shared-memory butterfly stores past the dynamic
window, which hard-faults under real occupancy. The native NVRTC kernel is the
default for that reason; select this path with ``TILEGEN_FHT_IMPL=tilelang``
for benchmarking or on stacks where the lowering is healthy.
"""
from __future__ import annotations

import torch
import tilelang
import tilelang.language as T

_KERNEL_CACHE: dict = {}


@tilelang.jit(execution_backend="nvrtc", out_idx=[1, 2])
def _make_kernel(M: int, K: int, group_size: int = 256, in_dtype: str = "float16"):
    assert group_size == 256, "the radix-4 specialization requires group_size=256"
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    # One row per CTA: the workspace-free absmax reduction is specialized to it.
    assert K % 128 == 0, f"K={K} must be a multiple of the 128-thread CTA"
    n_groups = K // group_size
    accum_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, K), in_dtype),
        XQ: T.Tensor((M, K), "int8"),
        Scale: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(M, threads=128) as bm:
            # +512 floats of deliberate slack: ptxas pipelines vectorized
            # LDS/STS past the buffer by up to one iteration on some stacks;
            # owning that page keeps the overrun inside this CTA's window.
            row_sh = T.alloc_shared((1, K + 512), accum_dtype)
            part_sh = T.alloc_shared((128,), accum_dtype)

            for k in T.Parallel(K):
                row_sh[0, k] = T.cast(X[bm, k], accum_dtype)
            T.sync_threads()

            # H256 = H4 kron H4 kron H4 kron H4: four radix-4 stages.
            for stage in T.unroll(4):
                stride = 1 << (stage * 2)
                for g, butterfly in T.Parallel(n_groups, 64):
                    low = butterfly % stride
                    high = butterfly // stride
                    base = g * group_size + high * (4 * stride) + low
                    a0 = row_sh[0, base]
                    a1 = row_sh[0, base + stride]
                    a2 = row_sh[0, base + 2 * stride]
                    a3 = row_sh[0, base + 3 * stride]
                    row_sh[0, base] = a0 + a1 + a2 - a3
                    row_sh[0, base + stride] = a0 + a1 - a2 + a3
                    row_sh[0, base + 2 * stride] = a0 - a1 + a2 + a3
                    row_sh[0, base + 3 * stride] = -a0 + a1 + a2 + a3
                T.sync_threads()

            # Row absmax via strided partials + an explicit shared tree fold.
            for t in T.Parallel(128):
                part_sh[t] = T.cast(0, accum_dtype)
            T.sync_threads()
            for kk in T.serial(K // 128):
                for t in T.Parallel(128):
                    part_sh[t] = T.max(
                        part_sh[t],
                        T.max(row_sh[0, kk * 128 + t], -row_sh[0, kk * 128 + t]),
                    )
            T.sync_threads()
            for step in T.serial(7):
                for t in T.Parallel(128):
                    if t < (64 >> step):
                        part_sh[t] = T.max(part_sh[t], part_sh[t + (64 >> step)])
                T.sync_threads()

            # Normalized H256 has entries +/-1/16; fold into the scale:
            # q = round((raw/16) / (raw_absmax/(16*127))) = round(raw / (scale*16)).
            for t in T.Parallel(128):
                if t == 0:
                    part_sh[0] = T.max(
                        part_sh[0] / T.cast(2032, accum_dtype),
                        T.cast(1e-30, accum_dtype),
                    )
                    Scale[bm] = part_sh[0]
            T.sync_threads()

            for k in T.Parallel(K):
                q = T.round(row_sh[0, k] / (part_sh[0] * T.cast(16, accum_dtype)))
                XQ[bm, k] = T.cast(
                    T.max(
                        T.min(q, T.cast(127, accum_dtype)),
                        T.cast(-128, accum_dtype),
                    ),
                    "int8",
                )

    return main


def quantize_convrot_tilelang(x, group_size: int = 256):
    """Fused rotate+quant via the tilelang lowering. Requires a CUDA 2-D tensor."""
    if group_size != 256:
        raise ValueError("the radix-4 specialization requires group_size=256")
    if x.ndim != 2:
        raise ValueError(f"expected 2-D input, got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("CUDA tensor required")
    if x.dtype == torch.float16:
        in_dtype = "float16"
    elif x.dtype == torch.bfloat16:
        in_dtype = "bfloat16"
    else:
        raise ValueError(f"unsupported activation dtype: {x.dtype}")

    m, k = x.shape
    key = (m, k, in_dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_kernel(m, k, group_size=group_size, in_dtype=in_dtype)
    kernel = _KERNEL_CACHE[key]
    return kernel(x.contiguous())
