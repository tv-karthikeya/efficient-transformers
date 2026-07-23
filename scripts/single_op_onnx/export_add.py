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
DTYPE = "fp16" # "bf16"

BATCH_SIZE = 4
DIM = 512

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
assert DTYPE in DTYPE_MAP, f"DTYPE must be one of {list(DTYPE_MAP.keys())}"

class AddModel(nn.Module):
    def forward(self, x, y):
        return torch.add(x,y)


model = AddModel().to(DTYPE_MAP[DTYPE]).eval()
dummy_x = torch.randn(BATCH_SIZE, DIM, dtype=DTYPE_MAP[DTYPE])
dummy_y = torch.randn(BATCH_SIZE, DIM, dtype=DTYPE_MAP[DTYPE])
out_path = f"/home/vtirumal/mainline_new/single_op_onnx_files/add/add_{DTYPE}.onnx" # update path

torch.onnx.export(
    model,
    (dummy_x, dummy_y),
    out_path,
    input_names=["input_x", "input_y"],
    output_names=["output"],
    dynamic_axes={
        "input_x": {0: "batch_size", 1: "dim"},
        "input_y": {0: "batch_size", 1: "dim"},
        "output":  {0: "batch_size", 1: "dim"},
    },
    opset_version=17,
)

print(f"Exported {out_path} → {os.path.abspath(out_path)}  (dtype={DTYPE})")
