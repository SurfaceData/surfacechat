import os
import torch
import yaml

from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from tqdm import tqdm
from typing import List, Optional

from surface_chat.serve.app_settings import app_settings
from surface_chat.serve.types import AdapterModel, ImageSettings

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


@router.get("/models", dependencies=[Depends(check_api_key)])
async def list_models() -> List[AdapterModel]:
    results = list(router.adapter_map.values())
    results.sort(key=lambda x: x.name)
    return results


class ImageGenerateRequest(BaseModel):
    id: str
    prompt: str
    negative_prompt: str
    lora: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 8.5
    profile_generate: bool = False
    profile_instruction: str = ""


class ImageGenerateResponse(BaseModel):
    image: str
    profile: str = ""


IMAGE_SIZES = [
    ("tiny", (64, 64)),
    ("small", (256, 256)),
    ("medium", (512, 512)),
]


@router.post("/generate", dependencies=[Depends(check_api_key)])
async def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    adapter = router.adapter_map[request.lora]
    router.base_pipeline.load_lora_weights(adapter.path())
    full_prompt = f"{request.prompt}, {adapter.keyword}"
    image = router.base_pipeline(
        prompt=full_prompt,
        negative_prompt=request.negative_prompt,
        height=request.height,
        width=request.width,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        # denoising_end=0.8,
        # output_type="latent",
    ).images[0]
    # image = router.refiner_pipeline(
    #     prompt=full_prompt,
    #     negative_prompt=request.negative_prompt,
    #     # Do significantly more steps on the refiner to clean things up.
    #     num_inference_steps=request.num_inference_steps * 10,
    #     guidance_scale=request.guidance_scale,
    #     denoising_start=0.8,
    #     image=image,
    # ).images[0]

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

    profile = ""
    return ImageGenerateResponse(
        image=os.path.join(app_settings.image_host, "results", full_file),
        profile=profile,
    )


def prepare_router():
    with open("image_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        image_settings = ImageSettings(**config)

    model_pack = image_settings.models[0]
    if model_pack.base.type == "pretrained":
        base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_pack.base.url,
            torch_dtype=torch.bfloat16 if app_settings.fp16 else torch.float32,
            variant="fp16" if app_settings.fp16 else None,
            use_safetensors=True,
        ).to(app_settings.device)
        base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            base_pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
    if model_pack.base.type == "single":
        base_pipeline = StableDiffusionXLPipeline.from_single_file(
            model_pack.base.path(),
            torch_dtype=torch.bfloat16 if app_settings.fp16 else torch.float32,
            use_safetensors=True,
        ).to(app_settings.device)
        base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            base_pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++",
        )
    # refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-refiner-1.0",
    #     text_encoder_2=base_pipeline.text_encoder_2,
    #     vae=base_pipeline.vae,
    #     torch_dtype=torch.bfloat16 if app_settings.fp16 else torch.float32,
    #     variant="fp16" if app_settings.fp16 else None,
    #     use_safetensors=True,
    # ).to(app_settings.device)
    # refiner_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    #     refiner_pipeline.scheduler.config,
    #     use_karras_sigmas=True,
    #     algorithm_type="sde-dpmsolver++",
    # )

    adapter_map = {}
    for model_pack in image_settings.models:
        for adapter in tqdm(model_pack.adapters, desc="Loading adapters..."):
            base_pipeline.load_lora_weights(adapter.path())
            adapter_map[adapter.name] = adapter

    router.adapter_map = adapter_map
    router.base_pipeline = base_pipeline
    # router.refiner_pipeline = refiner_pipeline
    return router
