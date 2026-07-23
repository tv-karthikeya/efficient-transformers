# Single-Op ONNX Export Scripts

Scripts to export individual operator ONNX models for benchmarking on Qualcomm AI 100.
Each script is self-contained — edit the globals at the top and run directly.

For a compact bf16/fp16 operator-by-operator view of the PyTorch code, export parameters,
and dynamic axes, see `OPERATOR_EXPORT_SUMMARY.md`.

For a focused explanation of MoE inference and int8 quantization, see
`MOE_INFERENCE_QUANTIZATION.md`.

For a visual token-level MoE walkthrough with continuous batching examples, open
`MOE_TOKEN_LEVEL_EXPLAINER.html` in a browser.

## bf16/fp16 Operator Export Summary

| Operator | PyTorch op | Supported dtypes | Trace input shape | Dynamic axes |
|---|---|---|---|---|
| Gemm | `nn.Linear(K, N, bias=True)` | `bf16`, `fp16` | `(M, K)` | `input[0]`, `output[0]` -> `M` |
| Add | `torch.add(x, y)` | `bf16`, `fp16` | `(BATCH_SIZE, DIM)` for both inputs | `input_x[0]`, `input_y[0]`, `output[0]` -> `batch_size`; `input_x[1]`, `input_y[1]`, `output[1]` -> `dim` |
| ReduceSum | `torch.sum(x, dim=-1, keepdim=True)` | `bf16`, `fp16` | `(BATCH_SIZE, DIM_SIZE)` | `input[0]`, `output[0]` -> `batch_size`; `input[1]`, `output[1]` -> `dim_size` |
| Softmax | `nn.Softmax(dim=-1)` | `bf16`, `fp16` | `(1, 1024)` | `input[0]`, `output[0]` -> `batch_size`; `input[1]`, `output[1]` -> `dim_size` |
| LayerNorm | `nn.LayerNorm(features=DIM_SIZE, eps=1e-5)` | `bf16`, `fp16` | `(BATCH_SIZE, DIM_SIZE)` | `input[0]`, `output[0]` -> `batch_size`; `input[1]`, `output[1]` -> `dim_size` |

---

## Scripts Overview

### 1. `export_gemm_only.py` — Single Gemm

| Property | Value |
|---|---|
| ONNX operator | `Gemm` |
| PyTorch class | `nn.Linear` (bf16/fp16), `GemmModelInt8` (int8) |
| Dynamic axes | `input[0]` → `M` |
| Fixed dims | `K`, `N` |

**Globals:**
```python
DTYPE = "bf16"      # "bf16" | "fp16" | "int8"
M, K, N = 4, 4096, 4096
```

**int8 note:** Uses weight-only int8 (`GemmModelInt8`): weights stored as `int8`, dequantized
via `Cast → Mul` before `Gemm`. Avoids `quantized::linear_dynamic` which is not ONNX-exportable.

**Output:** `gemm_only_{DTYPE}.onnx`

**Benchmark shapes:**
- M: `[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]`
- KN: `[4096×4096, 8192×8192, 8192×1024, 1024×8192]`

---

### 2. `export_gemm_only_all_comb.py` — Gemm All Combinations

| Property | Value |
|---|---|
| ONNX operator | `Gemm` |
| PyTorch class | `GemmModel` (bf16/fp16), `GemmModelInt8` (int8) |
| Dynamic axes | `input[0]` → `M` |
| Fixed dims | `K`, `N` |

**Globals:**
```python
M = 4               # trace shape only — M is dynamic
DTYPES = ["bf16", "fp16", "int8"]
KN_VALUES = [[4096, 4096], [8192, 8192], [8192, 1024], [1024, 8192]]
OUTPUT_DIR = "gemm_exports"
```

Generates `3 dtypes × 4 KN = 12` files organized as:
```
gemm_exports/
  bf16/gemm_K4096_N4096.onnx
  fp16/gemm_K4096_N4096.onnx
  int8/gemm_K4096_N4096.onnx
  ...
```

---

### 3. `export_softmax.py` — Softmax

| Property | Value |
|---|---|
| ONNX operator | `Softmax` |
| PyTorch class | `nn.Softmax(dim=-1)` |
| Dynamic axes | `input[0]` → `batch_size`, `input[1]` → `dim_size` |

**Globals:**
```python
DTYPE = "bf16"      # "bf16" | "fp16"
BATCH_SIZE = 1      # trace shape only
DIM_SIZE = 1024     # trace shape only
```

**Output:** `softmax_{DTYPE}.onnx`

**Benchmark shapes:**
- `dim_size`: `[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]`

---

### 4. `export_add.py` — Add

| Property | Value |
|---|---|
| ONNX operator | `Add` |
| PyTorch class | `torch.add(x, y)` |
| Dynamic axes | `input_x[0]`, `input_y[0]` → `batch_size`; `[1]` → `dim` |

**Globals:**
```python
DTYPE = "bf16"      # "bf16" | "fp16"
BATCH_SIZE = 4      # trace shape only
DIM = 8192          # trace shape only
```

**Output:** `add_{DTYPE}.onnx`

**Benchmark shapes:**
- M: `[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]`
- Fixed dim: `8192`

---

### 5. `export_reduce_sum.py` — ReduceSum

| Property | Value |
|---|---|
| ONNX operator | `ReduceSum` |
| PyTorch class | `torch.sum(dim=-1, keepdim=True)` |
| Dynamic axes | `input[0]` → `batch_size`, `input[1]` → `dim_size` |

**Globals:**
```python
DTYPE = "bf16"      # "bf16" | "fp16"
BATCH_SIZE = 1      # trace shape only
DIM_SIZE = 1024     # trace shape only
```

**Output:** `reduce_sum_{DTYPE}.onnx`

**Benchmark shapes:**
- `dim_size`: `[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]`

---

### 6. `export_layernorm.py` — LayerNorm

| Property | Value |
|---|---|
| ONNX operator | `LayerNormalization` |
| PyTorch class | `nn.LayerNorm` |
| Dynamic axes | `input[0]` → `batch_size`, `input[1]` → `dim_size` |

**Globals:**
```python
DTYPE = "bf16"      # "bf16" | "fp16"
BATCH_SIZE = 1024   # trace shape only
DIM_SIZE = 1024     # trace shape only
```

**Output:** `layernorm_{DTYPE}.onnx`

**Benchmark shapes:**
- `dim_size`: `[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]`

---

### 7. INT8 Output Quantized MatMul / MoE Exports

These two scripts model int8 input tensors, int8 weight tensors, and int8 output tensors. The quantization scales
stay `float32` because scales are real-valued conversion factors, not quantized data tensors.

| Script | Compute | Output dtype | Main inputs | Dynamic axes |
|---|---|---|---|---|
| `export_quant_matmul_int8_out.py` | ONNX `QLinearMatMul` int8 × int8 → int8 | int8 | `mat_a`, `mat_b`, `scale_a`, `scale_b`, `output_scale` | `mat_a[0]`, `output[0]` → `num_tokens` |
| `export_moe_quant_group_gemm.py` | routed int8 × int8 expert matmul → int8 requantized output | int8 | `x`, `expert_weights`, `expert_ids`, `scale_x`, `scale_w`, `output_scale` | `x[0]`, `expert_ids[0]`, `output[0]` → `num_tokens` |

**Common int8 quantization rule:**

```python
# Tensor data is int8.
mat_a = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.int8)
mat_b = torch.zeros(HIDDEN_SIZE, NUM_EXPERTS, dtype=torch.int8)

# Quantization scales are float32, not int8.
scale_x = torch.tensor(1.0, dtype=torch.float32)
scale_w = torch.tensor(1.0, dtype=torch.float32)
output_scale = torch.tensor(1.0, dtype=torch.float32)
```

The output tensor is int8 because the ONNX output type is int8 or the PyTorch model explicitly returns `.to(torch.int8)`.
Do not make `output_scale` int8; that would quantize the scale value itself, not the output tensor.

---

### 7a. `export_quant_matmul_int8_out.py` — QLinearMatMul with INT8 Output

| Property | Value |
|---|---|
| ONNX operator | `QLinearMatMul` |
| Explicit Cast nodes | none |
| Input dtype | `mat_a`: int8, `mat_b`: int8 |
| Scale dtype | `scale_a`: float32, `scale_b`: float32, `output_scale`: float32 |
| Output dtype | int8 |
| Dynamic axes | `mat_a[0]`, `output[0]` → `num_tokens` |

**Globals:**
```python
NUM_TOKENS = "num_tokens"  # symbolic dynamic axis
HIDDEN_SIZE = 8192
NUM_EXPERTS = 128
OUT_PATH = "quant_matmul_int8_out.onnx"
```

**Inputs:**

| Input | Shape | Dtype |
|---|---|---|
| `mat_a` | `(num_tokens, 8192)` | int8 |
| `mat_b` | `(8192, 128)` | int8 |
| `scale_a` | scalar | float32 |
| `scale_b` | scalar | float32 |
| `output_scale` | scalar | float32 |

**QLinearMatMul formula:**
```python
real_a = (mat_a - zero_point_a) * scale_a
real_b = (mat_b - zero_point_b) * scale_b
real_output = real_a @ real_b
output = quantize_to_int8(real_output, output_scale, zero_point_output)
```

**Notes:**
- The graph uses `QLinearMatMul` directly to avoid `Cast → MatMul → Round → Clip → Cast`.
- Zero-points are constant int8 initializers set to zero for symmetric signed int8 quantization.
- `mat_b` is shaped `(hidden_size, num_experts)` because `QLinearMatMul` computes `A[M, K] @ B[K, N]`.
- Only `num_tokens` is dynamic; `hidden_size=8192` and `num_experts=128` are fixed by export.

**Output:** `quant_matmul_int8_out.onnx`

---


### 8. `export_moe_quant_group_gemm.py` — MoE Grouped Quantized GEMM with INT8 Output

| Property | Value |
|---|---|
| ONNX operator pattern | `Gather → Cast → MatMul → Mul → Div → Round → Clip → Cast` per routed token |
| PyTorch class | `MoeQuantGroupGemm` |
| Input dtype | `x`: int8, `expert_weights`: int8, `expert_ids`: int32 |
| Scale dtype | `scale_x`: float32, `scale_w`: float32, `output_scale`: float32 |
| Output dtype | int8 |
| Routing modeled | top-1 expert ID per token |
| Dynamic axes | `x[0]`, `expert_ids[0]`, `output[0]` → `num_tokens` |

**Globals:**
```python
NUM_TOKENS = 48         # trace shape only — num_tokens is dynamic
HIDDEN_SIZE = 8192
NEW_HIDDEN_SIZE = 8192
NUM_EXPERTS = 128
OUT_PATH = "moe_quant_group_gemm_int8_out.onnx"
```

**Inputs:**

| Input | Shape | Dtype | Meaning |
|---|---|---|---|
| `x` | `(num_tokens, hidden_size)` | int8 | flattened token activations |
| `expert_weights` | `(num_experts, new_hidden_size, hidden_size)` | int8 | all expert weight matrices |
| `expert_ids` | `(num_tokens,)` | int32 | routed expert index for each token |
| `scale_x` | scalar | float32 | activation scale |
| `scale_w` | scalar | float32 | expert weight scale |
| `output_scale` | scalar | float32 | output int8 quantization scale |

**Per-token routed formula:**
```python
for i in range(num_tokens):
    w = expert_weights[expert_ids[i]]
    acc = x[i].int32 @ w.int32.T
    real_output = acc.float32 * scale_x * scale_w
    quant_output = round(real_output / output_scale)
    output[i] = clamp(quant_output, -128, 127).int8
```

**Token-level meaning:**
- `num_tokens` is the flattened active token list seen by the MoE layer.
- Tokens may come from one prompt, multiple prompts, decode, prefill, or continuous batching.
- `expert_ids[i]` chooses which expert matrix handles token `x[i]`.
- The output row remains aligned with the original token row index `i`.

**Benchmark config:**
- `ep_size=8`, `num_experts=128`, `topk=128` are benchmark context values, but this ONNX graph itself uses top-1 `expert_ids`.
- `sp_size=1`, `hidden_size=8192`, `new_hidden_size=8192`.
- dtype variant: `int8.int8.int8.int8`.

**Output:** `moe_quant_group_gemm_int8_out.onnx`

---

## Compile example (Qualcomm AI 100)

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=gemm_only_bf16.onnx \
  -onnx-define-symbol=M,8 \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=16 \
  -aic-binary-dir=./gemm_bf16_qpc
```
