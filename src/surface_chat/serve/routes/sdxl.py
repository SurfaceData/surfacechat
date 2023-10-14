import aiohttp
import asyncio
import httpx
import json
import os
import shortuuid
import torch

from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from typing import Any, Dict, Generator, List, Optional, Union

from surface_chat.serve.app_settings import app_settings

router = APIRouter(
    prefix="/sdxl",
    tags=["sdxl"],
)

get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    return None


class ImageGenerateRequest(BaseModel):
    id: str
    prompt: str
    negative_prompt: str
    lora: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 8.5


class ImageGenerateResponse(BaseModel):
    image: str


IMAGE_SIZES = [
    ("tiny", (64, 64)),
    ("small", (256, 256)),
    ("medium", (512, 512)),
]


@router.post("/generate", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: ImageGenerateRequest) -> ImageGenerateResponse:
    lora_model = LORA_MAP[request.lora]
    router.base_pipeline.load_lora_weights(lora_model.path)
    full_prompt = f"{lora_model.keyword}, {request.prompt}"
    image = router.base_pipeline(
        prompt=full_prompt,
        negative_prompt=request.negative_prompt,
        height=request.height,
        width=request.width,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        denoising_end=0.8,
        output_type="latent",
    ).images
    image = router.refiner_pipeline(
        prompt=full_prompt,
        negative_prompt=request.negative_prompt,
        # Do significantly more steps on the refiner to clean things up.
        num_inference_steps=request.num_inference_steps * 10,
        guidance_scale=request.guidance_scale,
        denoising_start=0.9,
        image=image,
    ).images[0]

    full_output_folder = f"{app_settings.image_basedir}/results"
    full_file = f"{request.id}_full.png"
    image_path = os.path.join(full_output_folder, full_file)
    image.save(
        image_path,
        compress_level=4,
    )
    for suffix, new_size in IMAGE_SIZES:
        resized_image = image.resize(new_size)
        resized_file = f"{request.id}_{suffix}.png"
        resized_image.save(
            os.path.join(full_output_folder, resized_file),
            compress_level=4,
        )
    return ImageGenerateResponse(
        image=os.path.join(app_settings.image_host, "results", full_file)
    )


class LoraModel(BaseModel):
    name: str
    path: str
    keyword: str


LORA_MAP = {
    "by-makoto-shinkai": LoraModel(
        name="by-makoto-shinkai",
        keyword="by Makoto Shinkai",
        path=f"{app_settings.image_basedir}/models/loras/sdxl_1.0_lora-by-makoto-shinkai.safetensors",
    ),
}


def prepare_router():
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if app_settings.fp16 else torch.float32,
        variant="fp16" if app_settings.fp16 else None,
        use_safetensors=True,
    ).to(app_settings.device)
    base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        base_pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
    )

    refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base_pipeline.text_encoder_2,
        vae=base_pipeline.vae,
        torch_dtype=torch.float16 if app_settings.fp16 else torch.float32,
        variant="fp16" if app_settings.fp16 else None,
        use_safetensors=True,
    ).to(app_settings.device)
    refiner_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        refiner_pipeline.scheduler.config,
        algorithm_type="sde-dpmsolver++",
    )

    for key in LORA_MAP:
        lora_model = LORA_MAP[key]
        print(lora_model.path)
        base_pipeline.load_lora_weights(lora_model.path)

    router.base_pipeline = base_pipeline
    router.refiner_pipeline = refiner_pipeline
    return router
