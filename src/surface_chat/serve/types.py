import os

from pydantic import BaseModel
from typing import List

from surface_chat.serve.app_settings import app_settings


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


class ImageSettings(BaseModel):
    models: List[ModelPack]
