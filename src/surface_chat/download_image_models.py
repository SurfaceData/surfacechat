import argparse
import os
import requests
import yaml

from tqdm import tqdm

from surface_chat.serve.app_settings import app_settings
from surface_chat.serve.types import ImageSettings


parser = argparse.ArgumentParser(
    description="Download stable diffusion checkpoints and adapters"
)

parser.add_argument(
    "config_file", type=argparse.FileType("r"), help="Path to a yaml configuration"
)

args = parser.parse_args()

with args.config_file as f:
    config = yaml.safe_load(f)
    image_settings = ImageSettings(**config)


def download(adapter):
    if os.path.exists(adapter.path()):
        return
    print(f"Downloading {adapter.name}")
    res = requests.get(adapter.url, params={"token": app_settings.civitai_key})
    if res.status_code != 200:
        print(res)
        return
    print(f"Saving {adapter.name}")
    with open(adapter.path(), "wb") as f:
        f.write(res.content)


for model_pack in image_settings.models:
    download(model_pack.base)
    for adapter in tqdm(model_pack.adapters, desc="Loading adapters..."):
        download(adapter)
