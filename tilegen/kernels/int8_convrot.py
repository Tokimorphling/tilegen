"""Fused ConvRot activation quantizer for INT8 ConvRot linear layers.

Implements the activation side of ConvRot quantization (group Hadamard rotation
+ row-wise INT8 quantization) as a single fused kernel:

    x [M, K] fp16/bf16
      -> (fused) rotate by group Hadamard (radix-4 butterfly) + row-wise int8
         => xq [M, K] int8, xscale [M] f32

Two implementations are provided, selected by the ``TILEGEN_FHT_IMPL`` env var:

* ``native`` (default): a hand-written NVRTC CUDA kernel where every shared
  byte is explicitly owned. Stable and fast across stacks; this is the path
  validated on Turing (RTX 2080 Ti) with a ~2x speedup over the upstream
  int8_convrot path.
* ``tilelang``: the tilelang lowering of the same radix-4 butterfly. Retained
  as the framework's intended path; on some toolchains (tilelang 0.1.12 /
  NVRTC 12.4 / sm75) its vectorized shared accesses overrun the dynamic window,
  so ``native`` is the default.

The weight side (offline rotation + per-channel int8) lives in
:mod:`tilegen.quant` and the ComfyUI nodes.
"""
from __future__ import annotations

import ctypes
import os

import torch

from ..runtime import nvrtc as _nvrtc
from ..runtime.device import usable_shared_bytes

KERNEL_GROUP_SIZE = 256


# ---------------------------------------------------------------------------
# Native NVRTC kernel (default, stable path)
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
#include <cuda_fp16.h>

#define K {K}
#define GROUP 256

extern "C" __global__ void __launch_bounds__(128)
fht_quant(const __half* __restrict__ X,
          signed char* __restrict__ XQ,
          float* __restrict__ Scale) {{
    extern __shared__ float row[];      // exactly K floats
    __shared__ float part[128];

    const long long m = blockIdx.x;
    const __half* xrow = X + m * (long long)K;

    for (int k = threadIdx.x; k < K; k += 128)
        row[k] = __half2float(xrow[k]);
    __syncthreads();

    // H256 = H4 (kron) H4 (kron) H4 (kron) H4: four radix-4 stages per group.
    for (int stage = 0; stage < 4; ++stage) {{
        const int stride = 1 << (2 * stage);
        for (int b = threadIdx.x; b < K / 4; b += 128) {{
            const int g = b / 64;
            const int t = b % 64;
            const int low = t % stride;
            const int high = t / stride;
            const int base = g * GROUP + high * 4 * stride + low;
            const float a0 = row[base];
            const float a1 = row[base + stride];
            const float a2 = row[base + 2 * stride];
            const float a3 = row[base + 3 * stride];
            row[base]              =  a0 + a1 + a2 - a3;
            row[base + stride]     =  a0 + a1 - a2 + a3;
            row[base + 2 * stride] =  a0 - a1 + a2 + a3;
            row[base + 3 * stride] = -a0 + a1 + a2 + a3;
        }}
        __syncthreads();
    }}

    float local = 0.0f;
    for (int k = threadIdx.x; k < K; k += 128)
        local = fmaxf(local, fabsf(row[k]));
    part[threadIdx.x] = local;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {{
        if (threadIdx.x < s)
            part[threadIdx.x] = fmaxf(part[threadIdx.x], part[threadIdx.x + s]);
        __syncthreads();
    }}

    // Unnormalized H256 entries are +/-1; folding the 1/16 normalization and
    // the int8 range into one constant gives absmax / (16 * 127) = / 2032.
    const float scale = fmaxf(part[0] / 2032.0f, 1e-30f);
    if (threadIdx.x == 0)
        Scale[m] = scale;
    const float inv = 1.0f / (scale * 16.0f);
    for (int k = threadIdx.x; k < K; k += 128) {{
        float q = nearbyintf(row[k] * inv);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        XQ[m * (long long)K + k] = (signed char)q;
    }}
}}
"""

_MODULES: dict[int, tuple[object, object]] = {}


def _cuda_include_dir() -> str | None:
    """Locate a CUDA toolkit include dir with cuda_fp16.h, version-agnostic.

    The native kernel only needs ``cuda_fp16.h`` for the ``__half`` type, which
    NVRTC ships built-in on 12.x+; on older stacks the include dir is still
    searched via CUDA_HOME / CUDA_PATH and common default locations across all
    CUDA versions (11.x through 13.x).
    """
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root and os.path.isfile(os.path.join(root, "include", "cuda_fp16.h")):
            return os.path.join(root, "include")
    # Common default install locations, scanned across versions.
    if os.name == "nt":
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        for v in os.listdir(base) if os.path.isdir(base) else []:
            inc = os.path.join(base, v, "include")
            if os.path.isfile(os.path.join(inc, "cuda_fp16.h")):
                return inc
    else:
        for cand in ("/usr/local/cuda/include", "/usr/include"):
            if os.path.isfile(os.path.join(cand, "cuda_fp16.h")):
                return cand
    return None


def _compile(k_dim: int) -> bytes:
    # The native kernel is plain CUDA C; NVRTC >= 11 is enough to compile it
    # (unlike the tilelang templates, which need >= 12 for c++20).
    _nvrtc.require_nvrtc(min_version=(11, 0))
    from cuda.bindings import nvrtc

    source = _KERNEL_SOURCE.format(K=k_dim)
    err, program = nvrtc.nvrtcCreateProgram(
        source.encode(), f"fht_quant_{k_dim}.cu".encode(), 0, [], []
    )
    if int(err) != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
    major, minor = torch.cuda.get_device_capability()
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-default-device"]
    include = _cuda_include_dir()
    if include:
        options.append(f"-I{include}".encode())
    err = nvrtc.nvrtcCompileProgram(program, len(options), options)[0]
    if int(err) != 0:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(f"fht_quant NVRTC compile failed:\n{log.decode(errors='replace')}")
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
    if int(err) != 0:
        raise RuntimeError(f"nvrtcGetCUBINSize failed: {err}")
    cubin = b" " * cubin_size
    err = nvrtc.nvrtcGetCUBIN(program, cubin)[0]
    if int(err) != 0:
        raise RuntimeError(f"nvrtcGetCUBIN failed: {err}")
    nvrtc.nvrtcDestroyProgram(program)
    return cubin


def _get_function(k_dim: int):
    if k_dim in _MODULES:
        return _MODULES[k_dim][1]
    from cuda.bindings import driver

    cubin = _compile(k_dim)
    err, module = driver.cuModuleLoadData(cubin)
    if int(err) != 0:
        raise RuntimeError(f"cuModuleLoadData failed: {err}")
    err, function = driver.cuModuleGetFunction(module, b"fht_quant")
    if int(err) != 0:
        raise RuntimeError(f"cuModuleGetFunction failed: {err}")
    shared_bytes = k_dim * 4
    err = driver.cuFuncSetAttribute(
        function,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_bytes,
    )[0]
    if int(err) != 0:
        raise RuntimeError(f"cuFuncSetAttribute(shared={shared_bytes}) failed: {err}")
    # Keep the module alive for the process lifetime; unloading from GC-time
    # destructors while launches are in flight is a real hazard.
    _MODULES[k_dim] = (module, function)
    return function


def quantize_convrot_native(x: torch.Tensor, group_size: int = 256):
    """Fused rotate+quant via the hand-written NVRTC kernel."""
    if group_size != KERNEL_GROUP_SIZE:
        raise ValueError(f"the radix-4 specialization requires group_size={KERNEL_GROUP_SIZE}")
    if x.ndim != 2:
        raise ValueError(f"expected 2-D input, got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("CUDA tensor required")
    if x.dtype != torch.float16:
        # bf16 rows are converted; the rotation happens in fp32 either way.
        x = x.to(torch.float16)
    m, k_dim = x.shape
    if k_dim % 256:
        raise ValueError(f"K={k_dim} not divisible by group_size=256")
    from cuda.bindings import driver

    x = x.contiguous()
    xq = torch.empty(m, k_dim, dtype=torch.int8, device=x.device)
    scale = torch.empty(m, dtype=torch.float32, device=x.device)
    function = _get_function(k_dim)

    arg_values = (x.data_ptr(), xq.data_ptr(), scale.data_ptr())
    arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    err = driver.cuLaunchKernel(
        function,
        m, 1, 1,
        128, 1, 1,
        k_dim * 4,
        torch.cuda.current_stream().cuda_stream,
        (arg_values, arg_types),
        0,
    )[0]
    if int(err) != 0:
        raise RuntimeError(f"fht_quant launch failed: {err}")
    return xq, scale


# ---------------------------------------------------------------------------
# Public API + tilelang dispatch
# ---------------------------------------------------------------------------

def _impl_choice() -> str:
    return os.environ.get("TILEGEN_FHT_IMPL", "native").strip().lower()


def quantize_convrot_fht(x: torch.Tensor, group_size: int = 256):
    """Apply the radix-4 fused activation quantizer (FHT) to a 2-D tensor.

    Dispatches to the native NVRTC kernel by default. Set
    ``TILEGEN_FHT_IMPL=tilelang`` to use the tilelang lowering instead.
    """
    if _impl_choice() != "tilelang":
        return quantize_convrot_native(x, group_size)
    from ._tilelang_fht import quantize_convrot_tilelang

    return quantize_convrot_tilelang(x, group_size)


def can_use_fht(x: torch.Tensor, group_size: int, min_k: int = 0) -> bool:
    """Whether the FHT kernel is applicable to ``x``.

    Args:
        min_k: minimum K (feature dim) to bother with FHT; below it eager is
            already fast. 0 means always use FHT when otherwise supported.
    """
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if group_size != KERNEL_GROUP_SIZE or x.shape[-1] % 256:
        return False
    if x.shape[-1] < min_k:
        return False
    # The kernel holds one float32 row in dynamic shared memory.
    return x.shape[-1] * 4 <= usable_shared_bytes(x.device.index or 0, headroom=2048)


def quantize_convrot(x: torch.Tensor, group_size: int = 256, backend: str = "auto", min_k: int = 0):
    """Quantize ``x`` with ConvRot, dispatching FHT vs eager automatically.

    Args:
        backend: ``"auto"`` (FHT when supported), ``"fht"`` (force FHT),
            ``"eager"`` (pure torch fallback).
        min_k: FHT minimum K threshold for ``auto``/``fht``.
    """
    backend = backend.strip().lower()
    if backend in ("auto", "fht") and can_use_fht(x, group_size, min_k=min_k if backend == "auto" else 0):
        try:
            return quantize_convrot_fht(x, group_size)
        except Exception:
            # Fall back to eager; the eager path is always correct.
            pass
    return _eager_quantize(x, group_size)


def _eager_quantize(x: torch.Tensor, group_size: int):
    """Pure-torch ConvRot quantization (stable reference / fallback)."""
    from .hadamard import build_hadamard

    if x.ndim != 2:
        raise ValueError(f"expected 2-D input, got {tuple(x.shape)}")
    m, k = x.shape
    h = build_hadamard(group_size, x.device, x.dtype)
    n_groups = k // group_size
    xg = x.reshape(m, n_groups, group_size)
    rotated = torch.matmul(xg, h).reshape(m, k)
    scale = (rotated.abs().amax(dim=-1).float() / 127.0).clamp_min(1e-30)
    qdata = torch.round(rotated / scale[:, None].to(rotated.dtype)).clamp(-128, 127).to(torch.int8)
    return qdata, scale

