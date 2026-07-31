"""Three-way benchmark for int8_convrot linear at real Wan-14B shapes.

Compares, at each shape:
  * bf16 F.linear                 (dense baseline)
  * comfy_kitchen int8_convrot    (upstream, when available)
  * tilegen FHT int8_convrot      (this package, native NVRTC)
  * tilegen eager int8_convrot    (pure-torch fallback)

Reports rel-err vs bf16 alongside the timing.

Run::

    python -m benchmarks.bench_int8_convrot
"""
from __future__ import annotations

import sys
import time

import torch

from tilegen.kernels.hadamard import build_hadamard
from tilegen.kernels.int8_convrot import _eager_quantize, quantize_convrot_fht
from tilegen.quant.int8_ops import int8_matmul_dequant_chunked
from tilegen.quant.weight import quantize_convrot_weight


# (name, K, N) for one Wan-14B transformer layer's key linear ops.
SHAPES = [
    ("qkv_proj", 5120, 15360),
    ("o_proj", 5120, 5120),
    ("ffn_up", 5120, 13824),
    ("ffn_down", 13824, 5120),
]
M = 4096
G = 256
ITERS = 30
WARMUP = 8


def _sync():
    torch.cuda.synchronize()


def bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    _sync()
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms/iter


def fht_forward(x, wq, wscale, bias):
    xq, xscale = quantize_convrot_fht(x, group_size=G)
    return int8_matmul_dequant_chunked(xq, xscale, wq, wscale, bias, x.dtype)


def eager_forward(x, wq, wscale, bias):
    xq, xscale = _eager_quantize(x, group_size=G)
    return int8_matmul_dequant_chunked(xq, xscale, wq, wscale, bias, x.dtype)


def _have_comfy_kitchen() -> bool:
    try:
        import comfy_kitchen  # noqa: F401
        from comfy_kitchen.tensor import int8 as _ck  # noqa: F401  (registers op)
        return hasattr(torch.ops, "comfy_kitchen") and hasattr(
            torch.ops.comfy_kitchen, "int8_linear"
        )
    except Exception:
        return False


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    dtype = torch.float16
    h = build_hadamard(G, dev, dtype)
    have_ck = _have_comfy_kitchen()

    cap = torch.cuda.get_device_capability(0)
    print(
        f"device={torch.cuda.get_device_name(0)}  sm={cap[0]}.{cap[1]}  "
        f"torch={torch.__version__}  dtype={dtype}  M={M}  group={G}  "
        f"iters={ITERS}  comfy_kitchen={have_ck}"
    )
    hdr = (
        f"{'shape':<11}{'K':>6}{'N':>6}{'bf16':>9}{'ck_orig':>9}"
        f"{'tg_fht':>9}{'tg_eager':>9}{'fht/orig':>9}{'fht/eager':>10}"
        f"{'err_fht':>9}{'err_orig':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, K, N in SHAPES:
        x = torch.randn(M, K, device=dev, dtype=dtype) * 0.1
        w = torch.randn(N, K, device=dev, dtype=dtype) * 0.1
        bias = torch.randn(N, device=dev, dtype=dtype)
        wq, wscale = quantize_convrot_weight(w, group_size=G)

        ref = torch.nn.functional.linear(x, w, bias)
        t_bf16 = bench(lambda: torch.nn.functional.linear(x, w, bias))

        t_ck = float("nan")
        rel_orig = float("nan")
        if have_ck:
            def _ck_call():
                return torch.ops.comfy_kitchen.int8_linear(
                    x.contiguous(), wq.contiguous(), wscale.contiguous(), bias, 2, True, G
                )

            out_ck = _ck_call()
            t_ck = bench(_ck_call)
            rel_orig = (
                (out_ck.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
            ).item()

        out_fht = fht_forward(x, wq, wscale, bias)
        t_fht = bench(lambda: fht_forward(x, wq, wscale, bias))

        out_eager = eager_forward(x, wq, wscale, bias)
        t_eager = bench(lambda: eager_forward(x, wq, wscale, bias))

        rel_fht = (
            (out_fht.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
        ).item()
        sp_orig = (t_ck / t_fht) if have_ck and t_fht > 0 else float("nan")
        sp_eager = t_eager / t_fht if t_fht > 0 else float("nan")

        print(
            f"{name:<11}{K:>6}{N:>6}{t_bf16:>9.2f}{t_ck:>9.2f}{t_fht:>9.2f}"
            f"{t_eager:>9.2f}{sp_orig:>8.2f}x{sp_eager:>9.2f}x"
            f"{rel_fht:>9.2e}{rel_orig:>9.2e}"
        )


if __name__ == "__main__":
    sys.exit(main())
