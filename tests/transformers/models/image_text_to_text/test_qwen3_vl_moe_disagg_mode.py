# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
# NOTE : yet to be validated 

import copy
import json
import os

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.run_utils import ApiRunnerVlm
from QEfficient.utils.test_utils import load_vlm_model_from_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_models"]
model_config_dict = {model["model_name"]: model for model in multimodal_models}

NEW_GENERATION_TOKENS = 10


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _update_qwen3_vl_moe_retained_states(target_inputs: dict, source_outputs: dict, num_hidden_layers: int):
    for layer_idx in range(num_hidden_layers):
        target_inputs[f"past_key.{layer_idx}"] = source_outputs[f"past_key.{layer_idx}_RetainedState"]
        target_inputs[f"past_value.{layer_idx}"] = source_outputs[f"past_value.{layer_idx}_RetainedState"]


def _run_qwen3_vl_moe_disagg_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    messages: list,
    vision_session: QAICInferenceSession,
    lang_prefill_session: QAICInferenceSession,
    lang_decode_session: QAICInferenceSession,
    prefill_seq_len: int,
    generation_len: int,
    batch_size: int,
) -> np.ndarray:
    qwen_vl_utils = pytest.importorskip("qwen_vl_utils")
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=batch_size
    )

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )

    inputs = {name: np.array(value) for name, value in inputs.items()}
    vision_inputs = {
        name: value
        for name, value in inputs.items()
        if name
        in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
    }
    vision_inputs.update(
        {
            name: vision_inputs[name].astype("float16")
            for name in {"pixel_values", "image_masks"}
            if name in vision_inputs
        }
    )
    vision_outputs = vision_session.run(vision_inputs)

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    lang_inputs["image_idx"] = np.array([[0]])
    for vision_output_name in ("vision_embeds", "deepstack_features"):
        if vision_output_name in vision_outputs:
            lang_inputs[vision_output_name] = vision_outputs[vision_output_name]

    lang_prefill_session.set_buffers(vision_outputs)
    chunk_inputs = lang_inputs.copy()
    outputs = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
        ]
        outputs = lang_prefill_session.run(chunk_inputs)
        _update_qwen3_vl_moe_retained_states(
            chunk_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers
        )
        chunk_inputs["image_idx"] = outputs["image_idx_output"]

    generated_ids = [_get_next_token_ids(outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(batch_size, 1),
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
        "image_idx": outputs["image_idx_output"],
    }
    _update_qwen3_vl_moe_retained_states(decode_inputs, outputs, qeff_model.model.config.text_config.num_hidden_layers)
    if "vision_embeds_RetainedState" in outputs:
        decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]

    decode_out = lang_decode_session.run(decode_inputs)
    generated_ids.append(_get_next_token_ids(decode_out["logits"]))
    pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
    loop_decode_inputs = {
        "input_ids": generated_ids[-1].reshape(batch_size, 1),
        "position_ids": pos_id,
        "image_idx": decode_out["image_idx_output"],
    }
    _update_qwen3_vl_moe_retained_states(
        loop_decode_inputs, decode_out, qeff_model.model.config.text_config.num_hidden_layers
    )
    if "vision_embeds_RetainedState" in decode_out:
        loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

    for _ in range(generation_len - 2):
        decode_out = lang_decode_session.run(loop_decode_inputs)
        generated_ids.append(_get_next_token_ids(decode_out["logits"]))
        pos_id += 1
        _update_qwen3_vl_moe_retained_states(
            loop_decode_inputs, decode_out, qeff_model.model.config.text_config.num_hidden_layers
        )
        loop_decode_inputs.update(
            {
                "input_ids": generated_ids[-1].reshape(batch_size, 1),
                "position_ids": pos_id,
                "image_idx": decode_out["image_idx_output"],
            }
        )
        if "vision_embeds_RetainedState" in decode_out:
            loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

    return np.stack(generated_ids, axis=1)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_qwen3_vl_moe_disagg_generation_matches_hf(manual_cleanup):
    pytest.importorskip("qwen_vl_utils")
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    qwen_model_config = model_config_dict[model_name]
    batch_size = qwen_model_config["batch_size"]
    prefill_seq_len = qwen_model_config["prompt_len"]
    ctx_len = qwen_model_config["ctx_len"]
    image_url = qwen_model_config["img_url"]
    query = qwen_model_config["text_prompt"]
    model_type = qwen_model_config["model_type"]
    hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **qwen_model_config["additional_params"])
    hf_config.name_or_path = model_name
    hf_config.dtype = "float32"

    model_hf = load_vlm_model_from_config(hf_config)
    qeff_model = QEFFAutoModelForImageTextToText(
        copy.deepcopy(model_hf),
        kv_offload=True,
        config=model_hf.config,
        torch_dtype=torch.float32,
        layerwise=False,
        enable_proxy=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

    image = Image.open(requests.get(image_url, stream=True).raw)
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        for _ in range(batch_size)
    ]
    prompt = processor.apply_chat_template(messages[0], add_generation_prompt=True)
    hf_inputs = processor(images=image, text=prompt, return_tensors="pt")
    if "pixel_values" in hf_inputs:
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
    api_runner = ApiRunnerVlm(
        batch_size,
        processor,
        hf_config,
        image,
        messages[0],
        prompt,
        prefill_seq_len,
        ctx_len,
        NEW_GENERATION_TOKENS,
        qwen_model_config["num_layers"],
    )
    pytorch_hf_tokens = api_runner.run_vlm_hf_model_on_pytorch(model_hf, hf_inputs).detach().cpu().numpy()

    sessions = []
    try:
        vision_qpc_path = qeff_model.compile(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=1,
            mos=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
        )
        prefill_qpc_path = qeff_model.compile(
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            retain_full_kv=True,
            split_model_io=True,
            mos=1,
            aic_enable_depth_first=True,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            layerwise_window_size=1,
        )
        decode_qpc_path = qeff_model.compile(
            batch_size=batch_size,
            prefill_seq_len=1,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=True,
            mxint8_kv_cache=True,
            split_model_io=True,
            mos=1,
            aic_enable_depth_first=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            layerwise_window_size=1,
        )
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
        lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
        lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))
        sessions.extend([vision_session, lang_prefill_session, lang_decode_session])

        cloud_ai_100_tokens = _run_qwen3_vl_moe_disagg_generation(
            qeff_model=qeff_model,
            processor=processor,
            messages=messages,
            vision_session=vision_session,
            lang_prefill_session=lang_prefill_session,
            lang_decode_session=lang_decode_session,
            prefill_seq_len=prefill_seq_len,
            generation_len=NEW_GENERATION_TOKENS,
            batch_size=batch_size,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert (pytorch_hf_tokens == cloud_ai_100_tokens[0]).all(), (
        "Tokens don't match for HF PyTorch output and Qwen3-VL-MoE disaggregated QPC output"
    )
