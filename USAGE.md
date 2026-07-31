# 运行文档 (Usage Guide)

tilegen 的 INT8 ConvRot 加速 kernel 可以两种方式使用：作为独立 Python 库，或作为 ComfyUI custom node。本文档覆盖安装、运行、测试与基准，以及常见问题排查。

---

## 1. 环境要求

| 组件 | 要求 |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.4（需带 CUDA 的 wheel，自带 NVRTC） |
| GPU | NVIDIA，计算能力 >= 7.5（Turing 及以上） |
| cuda-python | >= 12.0（`pip install cuda-python`） |

可选依赖：

| 组件 | 作用 | 安装 |
|---|---|---|
| `tilelang` | 使用 tilelang lowering 路径（默认走 native NVRTC） | `pip install "tilelang>=0.1.12"` |
| `comfy-kitchen` | ComfyUI 集成 + cuBLASLt IMMA GEMM | 随 ComfyUI 安装 |
| `pytest` | 用 pytest 跑测试（可选，也支持独立运行） | `pip install pytest` |

**关于 NVRTC**：无需单独安装 CUDA Toolkit。NVRTC 随 torch wheel 分发（Windows 在 `torch/lib/nvrtc64_*.dll`，Linux 在 `torch/lib/libnvrtc.so*`）。tilegen 会自动定位并预加载它。详见 [NVRTC 说明](#4-nvrtc-说明)。

---

## 2. 安装

### 2.1 核心库（独立使用）

```bash
git clone git@github.com:Tokimorphling/tilegen.git
cd tilegen
pip install torch cuda-python
pip install -e .
```

验证安装：

```bash
python -m tests.test_smoke
```

期望输出：

```
  PASS  hadamard_is_normalized_and_orthogonal
  PASS  rotate_by_group_preserves_norm
  PASS  weight_quant_shape_and_dtype
  PASS  fht_vs_eager_numerics
  PASS  can_use_fht_thresholds
  PASS  end_to_end_linear_vs_bf16

6 passed, 0 failed
```

### 2.2 ComfyUI 集成

将 `comfyui/` 子目录作为 custom node 安装到 ComfyUI：

```bash
# Linux / macOS
ln -s /path/to/tilegen/comfyui  /path/to/ComfyUI/custom_nodes/tilegen

# Windows（需管理员权限的符号链接，或直接复制）
mklink /D  D:\content\ComfyUI\custom_nodes\tilegen  D:\codes\tilegen\comfyui
# 或直接复制目录
xcopy /E /I D:\codes\tilegen\comfyui  D:\content\ComfyUI\custom_nodes\tilegen
```

重启 ComfyUI，启动日志应出现：

```
tilegen ConvRot backend enabled: mode=auto  FHT min K=5120  temp=256 MiB
```

这表示后端已注册。此后任何调用 `int8_linear(convrot=True)` 的模型（如 SCAIL-2 int8_convrot）会自动走 tilegen 的 FHT kernel。

---

## 3. 运行

### 3.1 作为独立库

```python
import torch
from tilegen.kernels import quantize_convrot, quantize_convrot_fht
from tilegen.quant import quantize_convrot_weight, int8_matmul_dequant_chunked

# 离线权重量化（只做一次，可缓存）
x = torch.randn(4096, 5120, device="cuda", dtype=torch.float16) * 0.1
w = torch.randn(5120, 5120, device="cuda", dtype=torch.float16) * 0.1
wq, wscale = quantize_convrot_weight(w, group_size=256)

# 推理前向：激活量化 + INT8 GEMM + 反量化
xq, xscale = quantize_convrot(x, group_size=256, backend="auto")
out = int8_matmul_dequant_chunked(xq, xscale, wq, wscale, bias=None, out_dtype=torch.float16)
```

`backend` 参数：

| 值 | 行为 |
|---|---|
| `"auto"`（默认） | 宽 K 用 FHT（更快），小 K 用 eager torch |
| `"fht"` | 强制 FHT（主要用于基准测试） |
| `"eager"` | 纯 PyTorch 路径（无 JIT，兼容性最高） |

### 3.2 在 ComfyUI 工作流中

**模型内部层（透明加速）**：无需改工作流。只要后端注册成功，模型内部的 ConvRot 线性层自动走 FHT。

**手动接线节点**（用于测试 / 自定义算子）：

| 节点 | 作用 |
|---|---|
| `QuantizeConvRotWeight` | 离线权重量化（运行一次，缓存输出） |
| `Int8ConvRotLinear` | 前向：融合旋转+量化激活 → INT8 GEMM → 反量化 |
| `ConvRotTestTensors` | 生成确定性测试张量 |

典型接法：

```
weight (fp16) → QuantizeConvRotWeight → (wq, wscale)
x (fp16) ─────────────────────────────→ Int8ConvRotLinear → output (fp16)
                 wq, wscale ──────────→
```

### 3.3 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `TILEGEN_CONVROT_BACKEND` | `auto` | 后端选择：`auto`/`fht`/`eager`/`off` |
| `TILEGEN_FHT_MIN_K` | 5120 (Turing) / 8192 (其他) | FHT 最低 K 阈值；设 `0` 强制全部用 FHT |
| `TILEGEN_INT8_TEMP_MB` | 256 | INT8 GEMM 分块的临时显存上限（MiB） |
| `TILEGEN_FHT_IMPL` | `native` | FHT 实现：`native`(NVRTC) / `tilelang`(tilelang lowering) |

---

## 4. NVRTC 说明

### 4.1 为什么不用装 CUDA Toolkit

NVRTC（运行时编译）随 torch wheel 一起分发：

- **Windows**：`torch/lib/nvrtc64_120_0.dll`（cu124）/ `nvrtc64_130_0.dll`（cu13x）
- **Linux**：`torch/lib/libnvrtc.so.12`（manylinux wheel 同样打包）

tilegen 在 `tilegen/runtime/nvrtc.py` 里自动定位并预加载它，所以 `requirements` 只要 `torch` + `cuda-python`，零额外步骤。

### 4.2 定位优先级

1. torch 自带的 NVRTC（首选，与 torch 的 CUDA runtime 匹配）
2. `nvidia-cuda-nvrtc-cuXX` pip 包（`nvidia/cuda_nvrtc/{bin,lib}`）
3. 系统 CUDA Toolkit（`PATH` / `CUDA_HOME` / `LD_LIBRARY_PATH`）

### 4.3 CUDA 13 / Blackwell 自适应

定位完全版本无关：用 glob `nvrtc64_*.dll` / `libnvrtc.so*` 匹配，所以 cu13x 的 torch（自带 NVRTC 13）会被自动选中。目标架构 `--gpu-architecture=sm_{major}{minor}` 在运行时从 `torch.cuda.get_device_capability()` 取，Blackwell 的 sm100/sm120 由 NVRTC 13 处理，无需改代码。

**唯一硬约束**：NVRTC 不能编译比自己新的架构。若 torch 与 GPU 不匹配（例如 torch cu124 + Blackwell），`require_nvrtc()` 会报清晰错误，指引用户升级 torch。

### 4.4 两个路径的版本门槛

| 路径 | 最低 NVRTC | 原因 |
|---|---|---|
| native NVRTC kernel（默认） | >= 11.0 | 纯 CUDA C |
| tilelang 模板路径 | >= 12.0 | c++20 模板 |

---

## 5. 测试

```bash
# 独立运行（无需 pytest）
python -m tests.test_smoke

# 用 pytest（如已安装）
pytest -q tests/
```

测试覆盖：
- Hadamard 矩阵正交性与归一化
- 分组旋转保范性
- 权重量化的形状与 dtype
- FHT vs eager 数值一致性（int8 差 <= 1）
- FHT 适用性判定（阈值/group_size/dtype）
- 端到端 linear vs bf16 基准（rel-err < 5e-2）

---

## 6. 基准测试

```bash
python -m benchmarks.bench_int8_convrot
```

对比 bf16 F.linear、comfy_kitchen 原版 int8_convrot、tilegen FHT、tilegen eager，输出各 Wan-14B 单层 shape 的计时与加速比。

参考结果（RTX 2080 Ti / Turing / fp16 / M=4096 / group=256）：

```
shape        K     N    bf16   ck_orig  tg_fht  tg_eager  fht/orig
qkv_proj   5120 15360  12.76   22.11   10.55   11.88      2.10x
o_proj     5120  5120   4.29    8.17    3.57    4.46      2.29x
ffn_up     5120 13824  11.41   20.92   11.01   11.64      1.90x
ffn_down  13824  5120  12.92   22.58    8.68   11.63      2.60x
```

---

## 7. 排查

| 现象 | 排查 |
|---|---|
| 启动无 "backend enabled" 日志 | 检查 `TILEGEN_CONVROT_BACKEND` 是否为 `off`；确认 comfy_kitchen 已安装 |
| `NVRTC ... is too old` | 升级 torch（它捆绑匹配的 NVRTC）；tilelang 路径需 >= 12.0 |
| `fht_quant NVRTC compile failed` | 检查 GPU 计算能力 >= 7.5；确认 torch 带对应 CUDA 版本 |
| FHT 不触发，回退 eager | 检查 K 是否低于阈值（`TILEGEN_FHT_MIN_K`）；设 `0` 强制启用 |
| 与原版数值不一致 | 确认 group_size=256 且权重用 `quantize_convrot_weight` 离线量化 |
| Linux 下 NVRTC 未找到 | 确认 torch 为带 CUDA 的 wheel（非 CPU-only）；或装 `nvidia-cuda-nvrtc-cuXX` |
