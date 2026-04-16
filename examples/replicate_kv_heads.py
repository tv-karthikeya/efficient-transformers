# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: 1D-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers import AutoConfig, AutoProcessor, TextStreamer
from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers, undo_transformers_quantizers
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear
from QEfficient.utils._utils import login_and_download_hf_lm
import requests
import transformers
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
import time 
import numpy as np
from time import perf_counter


def duplicate_weights_for_linear_layer(
    layer: torch.nn.Module, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
):
    new_kv_heads = repeat * orig_kv_heads
    if isinstance(layer, (WQLinear_GEMM, QuantLinearGPTQ)):
        if head_dim % 8 != 0:
            raise ValueError(
                f"the value head_dim={head_dim} is not divisible by 8 which is \
                                according to the assumption that model is 4-bit quantized."
            )
        if hidden_size % layer.group_size != 0:
            raise ValueError(
                f"The value of hidden_size={hidden_size} is not divisible by \
                            K_proj.group_size={layer.group_size}"
            )

        # Duplication of quantized weights
        layer.qweight.data = torch.repeat_interleave(
            layer.qweight.data.view(hidden_size, orig_kv_heads, head_dim // 8), repeat, 1
        ).view(hidden_size, (new_kv_heads * head_dim) // 8)
        # Duplication of quantized zero points
        layer.qzeros.data = torch.repeat_interleave(
            layer.qzeros.data.view(hidden_size // layer.group_size, orig_kv_heads, head_dim // 8),
            repeat,
            1,
        ).view(hidden_size // layer.group_size, (new_kv_heads * head_dim) // 8)
        # Duplication of quantization scales
        layer.scales.data = torch.repeat_interleave(
            layer.scales.data.view(hidden_size // layer.group_size, orig_kv_heads, head_dim),
            repeat,
            1,
        ).view(hidden_size // layer.group_size, new_kv_heads * head_dim)
        layer.out_features = layer.out_features * repeat

    elif isinstance(layer, FP8DeQuantLinear):
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
        ).view(new_kv_heads * head_dim, hidden_size)
        layer.weight_scale.data = torch.repeat_interleave(
            layer.weight_scale.data.view(orig_kv_heads, head_dim), repeat, 0
        ).view(new_kv_heads * head_dim, -1)

    else:
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
        ).view(new_kv_heads * head_dim, hidden_size)
        if layer.bias is not None:
            layer.bias.data = torch.repeat_interleave(layer.bias.data.view(orig_kv_heads, head_dim), repeat, 0).view(
                new_kv_heads * head_dim
            )


def replicate_kv_heads(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt: str = "My name is",
    repeat: int = 4,
    full_batch_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    hidden_size: Optional[int] = None,
):
    """
    Replicate the KV heads. The script performs the following steps:
    1. Runs inference with the original model.
    2. Replicates the KV heads.
    3. Runs inference on the modified model to validate the changes.
    4. Exports the modified model to ONNX format.

    ``Mandatory`` Args:
        :model_name (str): Model card name to use, default value as meta-llama/Meta-Llama-3-8B-Instruct.
        :prompt (str): Prompt to use for the model, default value as My name is
        :repeat (int): Factor to repeat key-value heads.
    ``Optional`` Args:
        :full_batch_size (int): Set full batch size to enable continuous batching mode, default is None.
        :num_hidden_layers (int): Number of hidden layers to use, default is None.
        :num_attention_heads (int): Number of attention heads, if not passed explicitly then will be picked from config.json.
        :hidden_size (int): Hidden size to use, if not passed explicitly then will be picked from config.json.

    """
    # Load the model and tokenizer
    # model_base_name = model_name.split("/")[-1]
    # Replace quantizers for loading Quantized AWQ/GPTQ models on CPU.
    replace_transformers_quantizers()
    # Prepare kwargs for model loading
    model_kwargs = {"attn_implementation": "eager"}

    if num_hidden_layers:
        model_kwargs["num_hidden_layers"] = num_hidden_layers
    # breakpoint()
    pretrained_model_name_or_path = login_and_download_hf_lm(model_name)
    config = AutoConfig.from_pretrained(model_name)
    # For Testing Purpose Only
    # config.vision_config.depth = 2
    config.text_config.num_hidden_layers = 25
    model = AutoModelForImageTextToText.from_pretrained(pretrained_model_name_or_path,torch_dtype=torch.float16,**model_kwargs,config=config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    # Undo the effect of replace_transformers_quantizers
    undo_transformers_quantizers()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # inputs = tokenizer(prompt, return_tensors="pt")

    # Generate original outputs and tokens
    # with torch.inference_mode():
    #     _ = model(**inputs)  # original output
    #     orig_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # Modify the number of key-value heads
    # breakpoint()
    orig_kv_heads = model.config.text_config.num_key_value_heads
    new_kv_heads = repeat * orig_kv_heads
    model.config.text_config.num_key_value_heads = orig_kv_heads #new_kv_heads

    print("Original KV heads:", orig_kv_heads)
    # print("Modified KV heads:", new_kv_heads)

    # Check if hidden size and number of attention heads are explicitly passed as arguments or not
    if num_attention_heads is None:
        num_attention_heads = model.config.text_config.num_attention_heads

    if hidden_size is None:
        hidden_size = model.config.text_config.hidden_size
    #NOTE : commented to try with 4 heads
    # # Update the model's attention layers with new key-value heads
    # for block in model.model.language_model.layers:
    #     attn = block.self_attn
    #     setattr(attn, "orig_kv_heads", orig_kv_heads)
    #     attn.num_key_value_heads = new_kv_heads
    #     attn.num_key_value_groups = num_attention_heads // new_kv_heads
    #     duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)
    #     duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)

    ## This won't work as attention_heads isn't divisible by num_kv_heads for repeat > 1, so we skip this inference run.
    # # Generate modified outputs and tokens
    # with torch.inference_mode():
    #     _ = model(**inputs)  # Modified output
    #     mod_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # # Print the original and modified token outputs
    # print("Original:", tokenizer.batch_decode(orig_tokens))
    # print("Modified:", tokenizer.batch_decode(mod_tokens))

    # if not torch.all(orig_tokens == mod_tokens):
    #     raise RuntimeError(
    #         "Something went wrong while duplicating KV heads weights, output token don't match after modification"
    #     )

    # Export the modified model
    q_model = QEFFAutoModelForImageTextToText(
        model,
        continuous_batching=(True if full_batch_size else False),
    )

    skip_vision = True #False
    if not skip_vision:
        print("........... with vision ...........")
        vision_qpc_path = q_model.compile(
            batch_size=1,
            prefill_seq_len=128,
            ctx_len=4096,
            height=354,
            width=536,
            num_cores=16,
            num_devices=8,
            mos=1,
            aic_enable_depth_first=True,
            split_retained_state_io=True,
            # prefill_only=True,
            # enable_chunking=True,
            skip_vision=skip_vision,
            skip_lang=True,
            use_onnx_subfunctions=True,
        )
        vision_session = QAICInferenceSession(vision_qpc_path)

    prefill_qpc_path = q_model.compile(
        batch_size=1,
        prefill_seq_len=128,
        ctx_len=4096,
        height=354,
        width=536,
        num_cores=16,
        num_devices=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,  # This should be used for disagg serving via VLLM
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
    )


    decode_qpc_path = q_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=4096,
        height=354,
        width=536,
        num_cores=16,
        num_devices=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,  # This should be used for disagg serving via VLLM
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
    )

    lang_prefill_session = QAICInferenceSession(prefill_qpc_path)
    if skip_vision:  # for only LLM with DA
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me about yourself."},
                ],
            },
        ]
    else:
        ### IMAGE + TEXT ###
        image_url = "https://picsum.photos/id/237/536/354"
        image = Image.open(requests.get(image_url, stream=True).raw)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Descibe all the colors seen in the image."},
                ],
            },
        ]


    messages = [messages] * 1

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = q_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=1)

    pad_token_id = 1
    input_len = inputs["attention_mask"].sum(1, keepdims=True)
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -128)  # ceil divide without float
    padded_len = num_chunks * 128  # Convert to a multiple of prompt_len
    generation_len = 100 #CTX_LEN - input_len.max()
    print(f"generation_len : {generation_len}")
    PREFILL_SEQ_LEN =128
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )

    for k, v in inputs.items():
        inputs[k] = np.array(v)

    vision_inputs = {
        k: v
        for k, v in inputs.items()
        if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
    }

    vision_inputs_fp16 = {"pixel_values", "image_masks"}
    vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

    vision_start = perf_counter()
    vision_outputs = {}
    if vision_inputs:
        vision_outputs = vision_session.run(vision_inputs)
    vision_end = perf_counter()
    if not skip_vision:
        vision_session.deactivate()
    print("_______________v____________________")

    lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask")
    else:
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

    lang_inputs["image_idx"] = np.array([[0]])

    if not skip_vision:
        lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]
        lang_inputs["deepstack_features"] = vision_outputs["deepstack_features"]

    # RUN prefill
    lang_start = perf_counter()
    lang_prefill_session.set_buffers(vision_outputs)
    all_outputs = []
    chunk_inputs = lang_inputs.copy()
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        outputs = lang_prefill_session.run(chunk_inputs)
        print(outputs["logits"],"\n>>>>>>> Range:", outputs["logits"].max() , outputs["logits"].min())
        print("___________________________________")
        for i in range(config.text_config.num_hidden_layers):
            chunk_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
            chunk_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
        chunk_inputs["image_idx"] = outputs["image_idx_output"]
    prefill_time = perf_counter() - lang_start + vision_end - vision_start
    print(f"Prefill time  :{prefill_time:.2f} secs")
    lang_prefill_session.deactivate()
    time.sleep(10)

    lang_decode_session = QAICInferenceSession(decode_qpc_path)

    all_outputs.append(np.argmax(outputs["logits"]))
    decode_inputs = {
        "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
    }

    for i in range(config.text_config.num_hidden_layers):
        decode_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        decode_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]

    decode_inputs["image_idx"] =  outputs["image_idx_output"]
    decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]
    decode_inputs["deepstack_features"] = outputs["deepstack_features_RetainedState"]

    st = perf_counter()
    decode_out = lang_decode_session.run(decode_inputs)
    print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
    loop_decode_inputs = {
        "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
        "position_ids": pos_id,
    }


    for i in range(config.text_config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
        loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]
    loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]
    loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]
    loop_decode_inputs["deepstack_features"] = decode_out["deepstack_features_RetainedState"]


    st = perf_counter()
    for i in range(generation_len - 2):
        decode_out = lang_decode_session.run(loop_decode_inputs)
        all_outputs.append(np.argmax(decode_out["logits"]))
        pos_id += 1
        for j in range(config.text_config.num_hidden_layers):
            loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
            loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]

        loop_decode_inputs.update(
            {
                "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
                "position_ids": pos_id,
            }
        )
    ft = perf_counter()
    print(all_outputs) #[785, 17133, 330, 11065, 23549, 678, 279, 7987, 3884, 304, 279, 2168, 1, 7952, 311, 387, 264]
    print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
    print(f"\noutput\n{tokenizer.decode(all_outputs)}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Modify and export a causal language model.")
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument("--prompt", type=str, default="My name is", help="Prompt to use for the model.")
    parser.add_argument("--repeat", type=int, default=4, help="Factor to repeat key-value heads.")
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
    parser.add_argument(
        "--num_hidden_layers",
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Number of hidden layers to use, default is None",
    )
    parser.add_argument(
        "--num_attention_heads",
        "--num-attention-heads",
        type=int,
        default=None,
        help="Number of attention heads, if not passed explicitly then will be picked from config.json",
    )
    parser.add_argument(
        "--hidden_size",
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size to use, if not passed explicitly then will be picked from config.json",
    )

    args = parser.parse_args()

    replicate_kv_heads(
        model_name=args.model_name,
        prompt=args.prompt,
        repeat=args.repeat,
        full_batch_size=args.full_batch_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_size=args.hidden_size,
    )