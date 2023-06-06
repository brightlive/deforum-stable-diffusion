#!/usr/bin/env python

import os
import shutil
import sys

# append project directory to path so predict.py can be imported
sys.path.append('.')
from helpers.model_load import download_model, load_model
from types import SimpleNamespace

from predict import MODEL_CACHE, MODELS

model_map = {
    "v1-5-pruned.ckpt": {
        'sha256': 'e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
        'requires_login': False,
        },
    "mdjrny-v4.ckpt": {
        'sha256': '02e37aad9f74f574808ad456043b89e8c6b24e22828743fcf002168f76493d9b',
        'url': "https://huggingface.co/prompthero/openjourney-v4/resolve/main/openjourney-v4.ckpt",
        'requires_login': False,
    },
}

for model in MODELS:
    MODEL_ID = model
    def Root():
        models_path = MODEL_CACHE #@param {type:"string"}
        configs_path = "configs" #@param {type:"string"}
        output_path = "outputs" #@param {type:"string"}
        mount_google_drive = True #@param {type:"boolean"}
        models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
        output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

        #@markdown **Model Setup**
        map_location = "cuda" #@param ["cpu", "cuda"]
        model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
        model_checkpoint =  MODEL_ID #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        custom_config_path = "" #@param {type:"string"}
        custom_checkpoint_path = "" #@param {type:"string"}
        return locals()

    root = Root()
    root = SimpleNamespace(**root)

    download_model(model_map, root)

