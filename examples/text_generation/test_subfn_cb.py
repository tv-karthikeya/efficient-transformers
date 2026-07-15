# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)
#   "EleutherAI/gpt-j-6b"
#   "bigcode/starcoder2-3b"
#   "tiiuae/falcon-7b"
 
model_name = "tiiuae/falcon-7b" # "EleutherAI/gpt-j-6b" #"microsoft/Phi-3-mini-4k-instruct" # "tiiuae/falcon-7b" # "bigcode/starcoder2-3b"
#"tiiuae/falcon-7b" #"EleutherAI/gpt-j-6b"
#
#"Qwen/Qwen2.5-3B-Instruct"
print(f">>>>>>>>>>> model_name : {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
print(config)
print("with trust_remote_code=False \n",config)
# config2 = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
# print("with trust_remote_code=True\n",config2)
breakpoint()
config.num_hidden_layers = 4
config.torch_dtype = torch.float16
use_dynamo = False
print(f">>>> use_dynamo : {use_dynamo} ")

if not hasattr(config, "max_position_embeddings"):
    config.max_position_embeddings = getattr(config, "n_positions", 2048)

if not hasattr(config, "ffn_hidden_size"):
    config.ffn_hidden_size = 4 * config.hidden_size

if not hasattr(config, "activation"):
    config.activation = getattr(config, "hidden_act", "gelu")

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, config=config, continuous_batching=True)
qeff_model.compile(
    prefill_seq_len=32, ctx_len=128, use_dynamo=use_dynamo, use_onnx_subfunctions=False, num_devices=4, # mxfp6_matmul=True, mxint8_kv_cache=True,
    full_batch_size=2,
)
print("compile done")
print("QEff Transformed Onnx Model Outputs(AIC Backend)")
output = qeff_model.generate(prompts=["My name is", "Explain Quantum Computing"], tokenizer=tokenizer, generation_len=100)
print(output.generated_ids)
