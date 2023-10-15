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
from yaml_settings_pydantic import BaseYamlSettings, YamlSettingsConfigDict

from surface_chat.serve.app_settings import app_settings

router = APIRouter(
    prefix="/sdxl",
    tags=["sdxl"],
)

get_bearer_token = HTTPBearer(auto_error=False)


class AdapterModel(BaseModel):
    name: str
    keyword: str = ""
    type: str
    url: str
    info: str

    def path(self):
        return os.path.join(
            app_settings.image_basedir,
            "models/loras",
            f"sdxl_1.0_{self.type}-{self.name}.safetensors",
        )


class ModelPack(BaseModel):
    name: str
    adapters: List[AdapterModel]


class ImageSettings(BaseYamlSettings):
    models: List[ModelPack]

    model_config = YamlSettingsConfigDict(yaml_files="image_config.yaml")


image_settings = ImageSettings()


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
    adapter = LORA_MAP[request.lora]
    router.base_pipeline.load_lora_weights(adapter.path())
    full_prompt = f"{adapter.keyword}, {request.prompt}"
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

    for model_pack in image_settings.models:
        for adapter in model_pack.adapters:
            base_pipeline.load_lora_weights(adapter.path())
            LORA_MAP[adapter.name] = adapter

    router.base_pipeline = base_pipeline
    router.refiner_pipeline = refiner_pipeline
    return router
