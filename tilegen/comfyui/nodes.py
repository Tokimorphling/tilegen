"""Standalone ComfyUI nodes for INT8 ConvRot quantization (any model).

These expose the ConvRot quantization path as standalone ComfyUI operations,
letting any model benefit from the tilegen FHT acceleration when using ConvRot
quantization. They are independent of the auto-registered backend in
:mod:`tilegen.comfyui.backend` (which accelerates model-internal layers).

Typical usage::

    weight (fp16) -> QuantizeConvRotWeight -> (wq, wscale)
    activation (fp16) -> Int8ConvRotLinear(wq, wscale) -> output (fp16)
"""
from __future__ import annotations

import logging
import os

import torch

from ..kernels.hadamard import build_hadamard
from ..kernels.int8_convrot import _eager_quantize, quantize_convrot_fht
from ..quant.int8_ops import int8_matmul_dequant_chunked
from ..quant.weight import quantize_convrot_weight
from ..runtime.device import usable_shared_bytes

LOGGER = logging.getLogger("tilegen.nodes")


class TileGenConvRotModelConfig:
    """Workflow-visible MODEL passthrough for TileGen runtime settings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["auto", "fht", "off"], {"default": "auto"}),
                "fht_min_k": ("INT", {"default": 8192, "min": 0, "max": 32768, "step": 256}),
                "fht_impl": (["native", "tilelang"], {"default": "native"}),
                "int8_temp_mb": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "diagnostics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "configure"
    CATEGORY = "model/optimization/tilegen"

    def configure(self, model, backend, fht_min_k, fht_impl, int8_temp_mb, diagnostics):
        os.environ["TILEGEN_CONVROT_BACKEND"] = backend
        os.environ["TILEGEN_FHT_MIN_K"] = str(max(0, int(fht_min_k)))
        os.environ["TILEGEN_FHT_IMPL"] = fht_impl
        os.environ["TILEGEN_INT8_TEMP_MB"] = str(max(16, int(int8_temp_mb)))
        os.environ["TILEGEN_DIAGNOSTICS"] = "1" if diagnostics else "0"
        LOGGER.warning(
            "tilegen workflow config: backend=%s min_k=%s impl=%s temp=%sMiB diagnostics=%s",
            backend, fht_min_k, fht_impl, int8_temp_mb, diagnostics,
        )
        return (model,)


class ConvRotTestTensors:
    """Deterministic tensors for the standalone ConvRot node test graph.

    V42 passes WANVIDEOMODEL objects through its chain, so it cannot provide the
    raw TENSOR inputs the two generic operators consume. This source node exists
    only to make such a test branch executable; it is not part of inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rows": ("INT", {"default": 32, "min": 1, "max": 4096, "step": 1}),
                "in_features": ("INT", {"default": 5120, "min": 4, "max": 32768, "step": 4}),
                "out_features": ("INT", {"default": 256, "min": 1, "max": 32768, "step": 1}),
                "group_size": ("INT", {"default": 256, "min": 4, "max": 1024, "step": 4}),
                "seed": ("INT", {"default": 20260726, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "dtype": (["float16", "bfloat16"], {"default": "float16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("x", "weight")
    FUNCTION = "create"
    CATEGORY = "quantization/convrot/testing"

    def create(self, rows, in_features, out_features, group_size, seed, dtype, device):
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by group_size={group_size}"
            )
        # Validate the group size before allocating (catches invalid powers).
        build_hadamard(group_size, "cpu", torch.float32)

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        resolved_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        generator = torch.Generator(device=resolved_device)
        generator.manual_seed(int(seed))
        x = torch.randn(
            (rows, in_features), generator=generator, device=resolved_device, dtype=resolved_dtype
        )
        weight = (
            torch.randn((out_features, in_features), generator=generator, device=resolved_device, dtype=resolved_dtype)
            * 0.02
        )
        return (x, weight)


class QuantizeConvRotWeight:
    """Offline weight quantization: rotate by group Hadamard + per-channel int8.

    Done once per weight and cacheable; the quantized weight is reused for all
    batches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight": ("TENSOR",),  # [N, K]
                "group_size": ("INT", {"default": 256, "min": 4, "max": 1024, "step": 4}),
            },
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("weight_quantized", "weight_scale")
    FUNCTION = "quantize"
    CATEGORY = "quantization/convrot"

    def quantize(self, weight: torch.Tensor, group_size: int):
        return quantize_convrot_weight(weight, group_size)


class Int8ConvRotLinear:
    """INT8 ConvRot linear: fused rotate+quant activation -> INT8 GEMM -> dequant.

    For K >= threshold the activation rotation+quantization uses the FHT NVRTC
    kernel; for smaller K or unsupported configurations it falls back to eager
    PyTorch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("TENSOR",),  # [M, K] or [*, K]
                "weight_quantized": ("TENSOR",),  # [N, K] int8
                "weight_scale": ("TENSOR",),  # [N]
                "group_size": ("INT", {"default": 256, "min": 4, "max": 1024, "step": 4}),
            },
            "optional": {
                "bias": ("TENSOR",),  # [N]
                "backend": (["auto", "fht", "eager"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "forward"
    CATEGORY = "quantization/convrot"

    def forward(self, x, weight_quantized, weight_scale, group_size, bias=None, backend="auto"):
        original_shape = tuple(x.shape)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        _, K = x2d.shape
        N = weight_quantized.shape[0]
        if K % group_size != 0:
            raise ValueError(f"K={K} must be divisible by group_size={group_size}")

        # Quantize activation (with backend selection).
        mode = os.environ.get("TILEGEN_CONVROT_BACKEND", backend).strip().lower()
        if mode in {"auto", "fht"} and self._can_use_fht(x2d, group_size, mode):
            try:
                xq, xscale = quantize_convrot_fht(x2d, group_size)
            except Exception as e:  # pragma: no cover
                print(f"[Int8ConvRotLinear] FHT failed, using eager: {e}")
                xq, xscale = _eager_quantize(x2d, group_size)
        else:
            xq, xscale = _eager_quantize(x2d, group_size)

        wq = weight_quantized.to(device=x.device, dtype=torch.int8)
        out = int8_matmul_dequant_chunked(
            xq, xscale, wq, weight_scale, bias, x.dtype
        )
        return (out.reshape(*original_shape[:-1], N),)

    @staticmethod
    def _can_use_fht(x: torch.Tensor, group_size: int, mode: str) -> bool:
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if group_size != 256 or x.shape[1] % 256:
            return False
        if mode == "auto":
            # Match the backend's default threshold (Turing 5120, else 8192).
            try:
                turing = torch.cuda.get_device_capability(x.device.index or 0) == (7, 5)
            except (AssertionError, RuntimeError):
                turing = False
            threshold = int(os.environ.get("TILEGEN_FHT_MIN_K", "5120" if turing else "8192"))
            if x.shape[1] < threshold:
                return False
        return x.shape[1] * 4 <= usable_shared_bytes(x.device.index or 0, headroom=2048)
