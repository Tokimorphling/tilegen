# tilegen

tilelang / NVRTC acceleration kernels for **DiT (diffusion transformer)** inference, plus a clean ComfyUI integration.

The first kernel is a fused **INT8 ConvRot** activation quantizer (group Hadamard rotation + row-wise INT8 quant), the op used by SCAIL-2 `int8_convrot` models. On Turing (RTX 2080 Ti) it is **~2x faster** than the upstream int8_convrot path and makes INT8 actually beat bf16 again.

```
shape        K     N    bf16   ck_orig  tg_fht  tg_eager  fht/orig
qkv_proj   5120 15360  12.76   22.11   10.55   11.88      2.10x
o_proj     5120  5120   4.29    8.17    3.57    4.46      2.29x
ffn_up     5120 13824  11.41   20.92   11.01   11.64      1.90x
ffn_down  13824  5120  12.92   22.58    8.68   11.63      2.60x
```
(M=4096, fp16, group=256, RTX 2080 Ti)

## Why

The upstream `comfy_kitchen` INT8 ConvRot path regresses below bf16 on Turing — INT8 GEMM (cuBLASLt IMMA) is fast, but the per-layer activation Hadamard rotation + quantization pays a big tensor-core matmul + conversion tax that eats the INT8 savings. tilegen fuses the rotation and quantization into a single NVRTC kernel that keeps the whole rotated row in shared memory, so the row-wise absmax (and thus the INT8 scale) is computed with no second pass and no HBM round-trip.

## Structure

```
tilegen/
├── tilegen/                 # core library, platform-independent
│   ├── runtime/             #   self-contained NVRTC bootstrap + device caps
│   ├── kernels/             #   hadamard + int8_convrot (native NVRTC + tilelang)
│   └── quant/               #   weight quant + INT8 GEMM/dequant (comfy_kitchen optional)
├── tilegen/comfyui/         # native ComfyUI backend + node implementation
├── comfyui/                 # thin drop-in custom_nodes loader
├── benchmarks/  tests/
```

## NVRTC: no separate install

NVRTC ships inside the torch wheel (`torch/lib/nvrtc64_*.dll` on Windows, `libnvrtc.so.*` on Linux). tilegen biases the loader toward that bundled copy, so **users do not install a CUDA toolkit**. Selection is version-agnostic — a torch built with cu13x (bundling NVRTC 13) is picked up automatically, and the target arch (`--gpu-architecture=sm_{major}{minor}`) is queried at runtime, so Blackwell (sm100/sm120) is handled by NVRTC 13 without code changes.

The only hard rule is physical: NVRTC cannot emit code for an arch newer than itself. If torch and the GPU mismatch, `require_nvrtc()` raises a clear message telling the user to upgrade torch.

## Install (core library)

```bash
pip install torch cuda-python
pip install -e .
```

## Install (ComfyUI integration)

Install the ComfyUI extra, then symlink or copy the top-level `comfyui/` loader
into `ComfyUI/custom_nodes/tilegen/` and restart ComfyUI:

```bash
pip install -e '.[comfyui]'
ln -s "$(pwd)/comfyui" /path/to/ComfyUI/custom_nodes/tilegen
```

The loader imports the installed `tilegen.comfyui` package, so its relative
imports resolve normally instead of escaping the custom-node package. On
startup the backend registers automatically:

```
tilegen selective FHT dispatch enabled: mode=auto FHT min K=5120 temp=256 MiB
```

Any model calling `int8_linear(convrot=True)` (SCAIL-2 int8_convrot, etc.) then dispatches through the FHT kernel. No workflow changes are needed — the acceleration is transparent to model-internal layers.

Backend selection via env var:

| `TILEGEN_CONVROT_BACKEND` | behavior |
|---|---|
| `auto` (default) | FHT for wide K, eager torch for small K |
| `fht` | force FHT for every supported K (benchmarking) |
| `eager` | pure-torch only (no-JIT compatibility) |
| `off` | do not register |

Override the FHT threshold with `TILEGEN_FHT_MIN_K` (0 = always use FHT).

## Standalone nodes

For manual / test wiring in a workflow:

- `QuantizeConvRotWeight` — offline weight quantization (run once, cache).
- `Int8ConvRotLinear` — forward: fused rotate+quant activation → INT8 GEMM → dequant.
- `ConvRotTestTensors` — deterministic test inputs.

## Tests & benchmarks

```bash
python -m pytest -q                  # correctness (needs CUDA)
python -m benchmarks.bench_int8_convrot   # timing vs bf16 + comfy_kitchen
```

## License

MIT.
