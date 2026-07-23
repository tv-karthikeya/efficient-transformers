# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export one ONNX file per (K, N, dtype) combination.

M is a dynamic axis — update the global M for the trace shape only.

Shapes
------
KN : [[4096, 4096], [8192, 8192], [8192, 1024], [1024, 8192]]

Dtypes
------
bf16  — weights and activations in bfloat16
fp16  — weights and activations in float16
int8  — weights stored as int8 + per-channel scale, dequantized before Gemm
        (Cast → Mul → Gemm, fully ONNX-compatible)

Output
------
gemm_exports/
  bf16_K4096_N4096.onnx
  fp16_K4096_N4096.onnx
  int8_K4096_N4096.onnx
  ...
"""

import os

import torch
import torch.nn as nn

M = 4  # example input for tracing only — M is dynamic at runtime
KN_VALUES = [[4096, 4096], [8192, 8192], [8192, 1024], [1024, 8192]]
DTYPES = ["bf16", "fp16", "int8"]
OUTPUT_DIR = "/home/vtirumal/mainline_new/single_op_onnx_files/gemm_exports"


class GemmModel(nn.Module):
    """Single nn.Linear → one ONNX Gemm node (bf16 / fp16)."""

    def __init__(self, K: int, N: int):
        super().__init__()
        self.fc = nn.Linear(K, N, bias=True)

    def forward(self, x):
        return self.fc(x)


class GemmModelInt8(nn.Module):
    """Weight-only INT8: weights stored as int8 + per-channel scale, dequantized before Gemm.
    Exports as Cast → Mul → Gemm, which is fully ONNX-compatible.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        self.weight_i8 = nn.Parameter(torch.randint(-128, 127, (N, K)).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(N, 1) * 0.01, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(N), requires_grad=False)

    def forward(self, x):
        w = self.weight_i8.float() * self.scale
        return x @ w.T + self.bias


DTYPE_TORCH = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "int8": torch.float32,  # activations float32; weights int8 dequantized before Gemm
}


def export_model(dtype: str, M: int, K: int, N: int, path: str) -> None:
    if dtype == "int8":
        model = GemmModelInt8(K, N).eval()
    else:
        model = GemmModel(K, N).to(DTYPE_TORCH[dtype]).eval()

    dummy = torch.randn(M, K, dtype=DTYPE_TORCH[dtype])

    torch.onnx.export(
        model,
        (dummy,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "M"}, "output": {0: "M"}},
        opset_version=17,
        do_constant_folding=True,
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dtype in DTYPES:
        dtype_dir = os.path.join(OUTPUT_DIR, dtype)
        os.makedirs(dtype_dir, exist_ok=True)
        for K, N in KN_VALUES:
            path = os.path.join(dtype_dir, f"gemm_K{K}_N{N}.onnx")
            export_model(dtype, M, K, N, path)
            print(f"Exported  dtype={dtype}  K={K:<6}  N={N:<6}  →  {os.path.abspath(path)}")

# /opt/qti-aic/exec/qaic-compile -m=/home/vtirumal/mainline_new/efficient-transformers/gemm_only_fp16.onnx  -onnx-define-symbol=M,8 -convert-to-fp16  -aic-hw  -aic-hw-version=ai100  -aic-num-cores=16 -aic-binary-dir=./fp16_gemma_qpc

if __name__ == "__main__":
    main()
