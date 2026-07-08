"""Model/processor loading — single place for the locked stack decisions.

bitsandbytes NF4 + SDPA + fp16 compute (see SPEC "Target hardware"); activations
are fp16 regardless of weight quantization, which is what the lens reads.
"""

from __future__ import annotations

import torch

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# ~250 visual tokens per 2-frame group: tokens = (H/28)*(W/28) after the 2x2
# spatial merge, so cap pixels at 250 * 28 * 28.
MAX_PIXELS = 250 * 28 * 28
DEFAULT_QUESTION = "Describe what happens in this video."


def load_model_and_processor(model_id: str = MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        attn_implementation="sdpa",
        dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=MAX_PIXELS)
    return model, processor


def decoder_layers(model):
    """The stack of decoder layers whose outputs the hooks capture."""
    return model.model.language_model.layers


def final_norm(model):
    return model.model.language_model.norm


def unembedding(model):
    return model.lm_head
