import os

import onnx
import torch
import torch.nn as nn

# ── Global config ─────────────────────────────────────────────────────────────
# dtype=int8, w_dtype=int8, accumulator=int32, output_dtype=int8
# num_tokens is dynamic at runtime; hidden_size and num_experts are fixed.
NUM_TOKENS = 48
HIDDEN_SIZE = 8192
NUM_EXPERTS = 128
OUT_PATH = "quant_matmul_int8_out.onnx"
OPSET = 17
# ─────────────────────────────────────────────────────────────────────────────


class QuantMatMulInt8Out(nn.Module):
    """
    INT8 x INT8 -> INT8 quantized matmul.

    Inputs:
        mat_a            : (num_tokens, hidden_size)  int8
        mat_b            : (num_experts, hidden_size) int8
        scale_a          : scalar float32, activation scale
        scale_b          : scalar float32, weight scale
        output_scale     : scalar float32, output quantization scale

    Compute:
        acc      = mat_a.int32 @ mat_b.int32.T
        dequant  = acc.float32 * scale_a * scale_b
        quant    = round(dequant / output_scale)
        output   = clamp(quant, -128, 127).int8
    """

    def forward(self, mat_a, mat_b, scale_a, scale_b, output_scale):
        acc = torch.matmul(mat_a.to(torch.int32), mat_b.to(torch.int32).t())
        dequant = acc.float() * (scale_a * scale_b)
        quant = torch.round(dequant / output_scale)
        return torch.clamp(quant, -128, 127).to(torch.int8)


model = QuantMatMulInt8Out().eval()

dummy_mat_a = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.int8)
dummy_mat_b = torch.zeros(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.int8)
dummy_scale_a = torch.tensor(1.0, dtype=torch.float32)
dummy_scale_b = torch.tensor(1.0, dtype=torch.float32)
dummy_output_scale = torch.tensor(1.0, dtype=torch.float32)

os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)) or ".", exist_ok=True)

torch.onnx.export(
    model,
    (dummy_mat_a, dummy_mat_b, dummy_scale_a, dummy_scale_b, dummy_output_scale),
    OUT_PATH,
    input_names=["mat_a", "mat_b", "scale_a", "scale_b", "output_scale"],
    output_names=["output"],
    dynamic_axes={
        "mat_a": {0: "num_tokens"},
        "output": {0: "num_tokens"},
    },
    opset_version=OPSET,
)

print(f"Exported {OUT_PATH} → {os.path.abspath(OUT_PATH)}")