# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import onnx
import torch
import torch.nn as nn

# ── Global config ─────────────────────────────────────────────────────────────
# dtype=int8, w_dtype=int8, accumulator=int32, output_dtype=int8
# ep_size=8, num_experts=128, topk=128 — fixed
# sp_size=1, hidden_size=8192, new_hidden_size=8192 — fixed
# num_tokens: [48, 8192] — dynamic at runtime
NUM_TOKENS = 48
HIDDEN_SIZE = 8192
NEW_HIDDEN_SIZE = 8192
NUM_EXPERTS = 128
OPSET = 17
# ─────────────────────────────────────────────────────────────────────────────

OUT_PATH = "moe_quant_group_gemm_int8_out.onnx"


class MoeQuantGroupGemm(nn.Module):
    """
    MoE grouped INT8 x INT8 -> INT8 quantized matmul.

    Inputs:
        x                : (num_tokens, hidden_size)                  int8
        expert_weights   : (num_experts, new_hidden_size, hidden_size) int8
        expert_ids       : (num_tokens,)                              int32, routed expert per token
        scale_x          : scalar float32, activation scale
        scale_w          : scalar float32, weight scale
        output_scale     : scalar float32, output quantization scale

    Compute (per token):
        For each token i routed to expert e:
            acc      = x[i].int32 @ expert_weights[e].int32.T
            dequant  = acc.float32 * scale_x * scale_w
            quant    = round(dequant / output_scale)
            output   = clamp(quant, -128, 127).int8
    """

    def forward(self, x, expert_weights, expert_ids, scale_x, scale_w, output_scale):
        num_tokens = x.shape[0]
        out = torch.zeros(num_tokens, expert_weights.shape[1], dtype=torch.float32)
        for i in range(num_tokens):
            w = expert_weights[expert_ids[i]]
            acc = torch.matmul(x[i].to(torch.int32), w.to(torch.int32).t())
            dequant = acc.float() * (scale_x * scale_w)
            out[i] = torch.round(dequant / output_scale)
        return torch.clamp(out, -128, 127).to(torch.int8)


model = MoeQuantGroupGemm().eval()

dummy_x = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.int8)
dummy_expert_weights = torch.zeros(NUM_EXPERTS, NEW_HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.int8)
dummy_expert_ids = torch.zeros(NUM_TOKENS, dtype=torch.int32)
dummy_scale_x = torch.tensor(1.0, dtype=torch.float32)
dummy_scale_w = torch.tensor(1.0, dtype=torch.float32)
# Quantization scales are real-valued, so they stay float32.
# The exported model output is int8 because forward() returns .to(torch.int8).
dummy_output_scale = torch.tensor(1.0, dtype=torch.float32)

os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)) or ".", exist_ok=True)

torch.onnx.export(
    model,
    (dummy_x, dummy_expert_weights, dummy_expert_ids, dummy_scale_x, dummy_scale_w, dummy_output_scale),
    OUT_PATH,
    input_names=["x", "expert_weights", "expert_ids", "scale_x", "scale_w", "output_scale"],
    output_names=["output"],
    dynamic_axes={
        "x":          {0: "num_tokens"},
        "expert_ids": {0: "num_tokens"},
        "output":     {0: "num_tokens"},
    },
    opset_version=OPSET,
)

print(f"Exported {OUT_PATH} → {os.path.abspath(OUT_PATH)}")