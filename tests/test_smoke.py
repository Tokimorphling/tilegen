"""Smoke + correctness tests for the tilegen core library (no ComfyUI needed).

Runs without comfy_kitchen installed; the INT8 matmul falls back to
``torch._int_mm``. Works with or without pytest::

    python -m tests.test_smoke          # standalone runner
    pytest -q tests/test_smoke.py       # if pytest is installed
"""
from __future__ import annotations

import sys

try:
    import torch
except ImportError:
    raise SystemExit("torch is required to run these tests")

if not torch.cuda.is_available():
    print("CUDA required for tilegen kernel tests; skipping.")
    raise SystemExit(0)

from tilegen.kernels.hadamard import build_hadamard, rotate_by_group
from tilegen.kernels.int8_convrot import (
    _eager_quantize,
    can_use_fht,
    quantize_convrot_fht,
)
from tilegen.quant.int8_ops import int8_matmul_dequant_chunked
from tilegen.quant.weight import quantize_convrot_weight


DEV = torch.device("cuda")
DT = torch.float16
G = 256


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()


def test_hadamard_is_normalized_and_orthogonal():
    h = build_hadamard(256, DEV, DT)
    eye = torch.matmul(h, h.t())
    assert _rel(eye, torch.eye(256, device=DEV, dtype=DT)) < 1e-2


def test_rotate_by_group_preserves_norm():
    x = torch.randn(8, 5120, device=DEV, dtype=DT) * 0.1
    h = build_hadamard(G, DEV, DT)
    xr = rotate_by_group(x, h, G)
    # Hadamard rotation is norm-preserving (1/sqrt(n) normalization).
    assert _rel(x.float().norm(dim=-1), xr.float().norm(dim=-1)) < 1e-2


def test_weight_quant_shape_and_dtype():
    w = torch.randn(5120, 5120, device=DEV, dtype=DT) * 0.1
    wq, wscale = quantize_convrot_weight(w, G)
    assert wq.shape == w.shape and wq.dtype == torch.int8
    assert wscale.shape == (5120,) and wscale.dtype == torch.float32


def test_fht_vs_eager_numerics():
    x = torch.randn(2048, 5120, device=DEV, dtype=DT) * 0.1
    xq_fht, xs_fht = quantize_convrot_fht(x, G)
    xq_eager, xs_eager = _eager_quantize(x, G)
    # INT8 results may differ by at most 1 (rounding near boundaries).
    assert (xq_fht.int() - xq_eager.int()).abs().max().item() <= 1
    assert (xs_fht - xs_eager).abs().max().item() < 1e-4


def test_can_use_fht_thresholds():
    x_cuda = torch.randn(8, 5120, device=DEV, dtype=DT)
    x_cpu = torch.randn(8, 5120, dtype=DT)
    assert can_use_fht(x_cuda, 256, min_k=0) is True
    assert can_use_fht(x_cpu, 256) is False  # not cuda
    assert can_use_fht(x_cuda, 128, min_k=0) is False  # wrong group size
    assert can_use_fht(x_cuda, 256, min_k=99999) is False  # below threshold


def test_end_to_end_linear_vs_bf16():
    M, K, N = 1024, 5120, 5120
    x = torch.randn(M, K, device=DEV, dtype=DT) * 0.1
    w = torch.randn(N, K, device=DEV, dtype=DT) * 0.1
    bias = torch.randn(N, device=DEV, dtype=DT)
    wq, wscale = quantize_convrot_weight(w, G)

    xq, xscale = quantize_convrot_fht(x, G)
    out = int8_matmul_dequant_chunked(xq, xscale, wq, wscale, bias, DT)
    ref = torch.nn.functional.linear(x, w, bias)
    # int8 convrot rel-err vs bf16 is ~1e-2 by construction.
    assert _rel(out, ref) < 5e-2


# --- standalone runner (no pytest required) ---
def _run_all():
    tests = [
        ("hadamard_is_normalized_and_orthogonal", test_hadamard_is_normalized_and_orthogonal),
        ("rotate_by_group_preserves_norm", test_rotate_by_group_preserves_norm),
        ("weight_quant_shape_and_dtype", test_weight_quant_shape_and_dtype),
        ("fht_vs_eager_numerics", test_fht_vs_eager_numerics),
        ("can_use_fht_thresholds", test_can_use_fht_thresholds),
        ("end_to_end_linear_vs_bf16", test_end_to_end_linear_vs_bf16),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
