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
DTYPE =  "fp16" # "bf16" 
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}

class SoftmaxModel(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


# Benchmark shape: (batch_size, dim_size) — both dims dynamic
# dim_size varies: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
model = SoftmaxModel(dim=-1).to(DTYPE_MAP[DTYPE]).eval()
dummy = torch.randn(1, 1024, dtype=DTYPE_MAP[DTYPE])
out_path = f"/home/vtirumal/mainline_new/single_op_onnx_files/softmax/softmax_{DTYPE}.onnx"

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
