# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import torch
import torch.nn as nn

DTYPE = "fp16" # "bf16"

# Trace shape only — both batch_size and dim_size are dynamic at runtime
BATCH_SIZE = 1024
DIM_SIZE = 1024

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
assert DTYPE in DTYPE_MAP, f"DTYPE must be one of {list(DTYPE_MAP.keys())}"


class LayerNormModel(nn.Module):
    def __init__(self, features: int = 1024, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(features, eps=eps)

    def forward(self, x):
        return self.ln(x)


# Benchmark dim_size varies: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
model = LayerNormModel(features=DIM_SIZE).to(DTYPE_MAP[DTYPE]).eval()
dummy = torch.randn(BATCH_SIZE, DIM_SIZE, dtype=DTYPE_MAP[DTYPE])
out_path = f"layernorm_{DTYPE}.onnx"

torch.onnx.export(
    model,
    (dummy,),
    out_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "dim_size"}, "output": {0: "batch_size", 1: "dim_size"}},
    opset_version=17,
)

print(f"Exported {out_path} → {os.path.abspath(out_path)}  (dtype={DTYPE})")
