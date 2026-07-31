"""Quantization helpers: weight-side ConvRot quant + INT8 GEMM/dequant.

The activation-side fused rotate+quant lives in :mod:`tilegen.kernels`; this
module covers the weight-side offline quantization and the INT8 matmul + scale
back path used by both the ComfyUI backend and the standalone nodes.

``comfy_kitchen`` is an optional dependency: when present, the GEMM uses its
``_int8_matmul_accumulate`` (cuBLASLt IMMA, with Turing padding); otherwise it
falls back to ``torch._int_mm``.
"""
from .int8_ops import int8_matmul_dequant_chunked
from .weight import quantize_convrot_weight

__all__ = ["int8_matmul_dequant_chunked", "quantize_convrot_weight"]
