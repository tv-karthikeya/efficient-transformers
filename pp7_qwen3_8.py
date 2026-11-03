#!/usr/bin/env python3
"""Compile the Qwen/Qwen3.8-27B language-prefill QPC."""

from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import SplitTensorsTransform


MODEL_ID = "Qwen/Qwen3.8-27B"

PREFILL_COMPILE_PARAMS = {
    # "lang_onnx_path":"/home/vtirumal/mainline/mdp_issue/Qwen3_5ForConditionalGeneration/Qwen3_5DecoderWrapper-2eb89d5747263146/Qwen3_5DecoderWrapper.onnx",
    "prefill_seq_len": 64,
    "ctx_len": 512,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 7,
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "aic_enable_depth_first": False,
    "mos": 1,
    "prefill_only": True,
    "use_onnx_subfunctions": True,
    "split_model_io": True,
    "retain_full_kv": True,
    "enable_chunking": True,
    "kv_cache_prefix": "vllmKvCache",
    "layerwise": False,
    "mdp_strategy": "intersection",
    "user_tiled": True,
    "allow_mxint8_mdp_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "img_size": 448,
    "skip_vision": True,
    "vision_size": 64,
    "mdp_num_partitions": 7,
}


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_ID)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        layerwise=False,
        trust_remote_code=False,
    )
    qeff_model.lang_model._onnx_transforms = [SplitTensorsTransform]
    qeff_model.lang_model.hash_params["split_tensors_external_data"] = True

    print(f"Compiling {MODEL_ID} with prefill parameters:")
    print(PREFILL_COMPILE_PARAMS)
    qpc_path = qeff_model.compile(**PREFILL_COMPILE_PARAMS)
    print(f"QPC path: {qpc_path}")
    # breakpoint()
    print(f">>>>>>>>>> onnx path {qeff_model.lang_model.onnx_path}")


if __name__ == "__main__":
    main()
