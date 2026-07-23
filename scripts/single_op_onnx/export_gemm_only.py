# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import torch
import torch.nn as nn

# ── Global config ─────────────────────────────────────────────────────────────
# DTYPE options: "bf16" | "fp16" | "int8"
DTYPE = "int8"
# ─────────────────────────────────────────────────────────────────────────────


class GemmModel(nn.Module):
    def __init__(self, K: int = 512, N: int = 512):
        super().__init__()
        self.fc = nn.Linear(K, N, bias=True)

    def forward(self, x):
        return self.fc(x)


class GemmModelInt8(nn.Module):
    """Weight-only INT8: weights stored as int8 + per-channel scale, dequantized before Gemm.
    Exports as Cast → Mul → Gemm, which is fully ONNX-compatible.
    """

    def __init__(self, K: int = 512, N: int = 512):
        super().__init__()
        # int8 weight and per-output-channel scale as plain tensors (ONNX-exportable)
        self.weight_i8 = nn.Parameter(torch.randint(-128, 127, (N, K)).to(torch.int8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(N, 1) * 0.01, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(N), requires_grad=False)

    def forward(self, x):
        # Dequantize: (N, K) int8 → float32
        w = self.weight_i8.float() * self.scale
        return x @ w.T + self.bias


# Benchmark shape: (M, K) — M varies, K/N fixed per config
# M:  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# KN: [4096, 4096], [8192, 8192], [8192, 1024], [1024, 8192]
M, K, N = 4, 4096, 4096

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "int8": torch.float32,  # activations float32; weights int8 dequantized before Gemm
}

assert DTYPE in DTYPE_MAP, f"DTYPE must be one of {list(DTYPE_MAP.keys())}"

if DTYPE == "int8":
    model = GemmModelInt8(K, N).eval()
else:
    model = GemmModel(K, N).to(DTYPE_MAP[DTYPE]).eval()

dummy = torch.randn(M, K, dtype=DTYPE_MAP[DTYPE])
out_path = f"gemm_only_{DTYPE}.onnx"

torch.onnx.export(
    model,
    (dummy,),
    out_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "M"}, "output": {0: "M"}},
    opset_version=17,
)

print(f"Exported {out_path} → {os.path.abspath(out_path)}  (M={M}, K={K}, N={N}, dtype={DTYPE})")
