import requests
import torch

from io import BytesIO
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import (
    process_images,
    load_image_from_base64,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from transformers import AutoTokenizer
from PIL import Image


class LLaVaGenerator:
    def __init__(self, device, model_path):
        self.device = device
        self.model_path = model_path
        self.dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        kwargs = {
            "device_map": "auto",
            "load_in_8bit": True,
            "torch_dtype": self.dtype,
        }
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(
            self.model.config, "mm_use_im_patch_token", True
        )
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=self.dtype)
        self.image_processor = vision_tower.image_processor

    def generate(self, image, instruction):
        images = process_images([image], self.image_processor, self.model.config)
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{instruction} ASSISTANT:"

        if type(images) is list:
            images = [image.to(self.device, dtype=self.dtype) for image in images]
        else:
            images = images.to(self.device, dtype=self.dtype)
        image_args = {"images": images}
        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        num_image_tokens = (
            prompt.count(replace_token) * self.model.get_vision_tower().num_patches
        )

        temperature = 1.0
        top_p = 1.0
        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        max_new_tokens = 1256
        stop_str = "</s>"
        do_sample = True if temperature > 0.001 else False
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        max_new_tokens = min(
            max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
        )

        result = self.model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args,
        )
        return self.tokenizer.batch_decode(
            result[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
