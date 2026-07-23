# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import torch
import torch.nn as nn

# DTYPE options: "bf16" | "fp16"
DTYPE = "bf16"

# Trace shape only — both batch_size and dim_size are dynamic at runtime
BATCH_SIZE = 1
DIM_SIZE = 1024

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
assert DTYPE in DTYPE_MAP, f"DTYPE must be one of {list(DTYPE_MAP.keys())}"


class ReduceSumModel(nn.Module):
    def __init__(self, dim: int = -1, keepdim: bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)


# Benchmark dim_size varies: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
model = ReduceSumModel(dim=-1, keepdim=True).to(DTYPE_MAP[DTYPE]).eval()
dummy = torch.randn(BATCH_SIZE, DIM_SIZE, dtype=DTYPE_MAP[DTYPE])
out_path = f"reduce_sum_{DTYPE}.onnx"

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
