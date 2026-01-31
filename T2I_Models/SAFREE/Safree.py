# from torchvision import transforms
import pandas as pd
import argparse
import torch
# import csv
import os
import json

from PIL import Image

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import DPMSolverMultistepScheduler

from T2I_Models.SAFREE.main_utils import Logger, read_json, dummy, horz_stack, vert_stack

from T2I_Models.SAFREE.models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from T2I_Models.SAFREE.models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline
from T2I_Models.SAFREE.models.modified_sld_pipeline import ModifiedSLDPipeline
from T2I_Models.SAFREE.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

import numpy as np
import pdb

SD_FUNCTIONS = {
    "std": ModifiedStableDiffusionPipeline,
    "std+xl": ModifiedStableDiffusionXLPipeline,
    # "std+v3": StableDiffusion3Pipeline,
    "esd": ModifiedStableDiffusionPipeline,
    "sld": ModifiedSLDPipeline,
}

SLD_CONFIGS = {
    "MAX": SafetyConfig.MAX,
    "STRONG": SafetyConfig.STRONG,
    "MEDIUM": SafetyConfig.MEDIUM,
    "WEAK": SafetyConfig.WEAK
}


def load_sd(pipeline_func, device, weight_dtype, unet_ckpt=None):
    scheduler = DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    pipe = pipeline_func.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        revision="fp16"
    )

    if unet_ckpt is not None:
        unet_weight = torch.load(unet_ckpt, map_location='cpu')
        try:
            pipe.unet.load_state_dict(unet_weight)
        except:
            pipe.unet.load_state_dict(unet_weight['unet'])
        print(f"ESD unet: {unet_ckpt} is loaded...")


    pipe.safety_checker = dummy
    pipe.image_encoder = dummy

    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.train()

    gen = torch.Generator(device=device)
    return pipe, gen

class Safree:
    def __init__(self, device):
        self.erase_id = "std"
        self.pipe, self.gen = load_sd(SD_FUNCTIONS[self.erase_id], device, torch.float32, None)

        ###
        freeu_hyp = "1.0-1.0-0.9-0.2"
        freeu_hyps = freeu_hyp.split('-')
        b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])
        self.re_attn_t = "-1,1001"
        self.sf_alpha = 0.01
        self.up_t = 10
        self.safe_config = None

        register_free_upblock2d(self.pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(self.pipe, b1=b1, b2=b2, s1=s1, s2=s2)
        ###
        self.negative_prompt_space = [
            "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
            "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
            "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
            "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
        ]
        self.negative_prompt = ", ".join(self.negative_prompt_space)

    def generate_images(self, prompt, image_num, seed=666):
        imgs = self.pipe(
            prompt,
            num_images_per_prompt=image_num,
            guidance_scale=7.5,
            num_inference_steps=50,
            negative_prompt=self.negative_prompt,
            negative_prompt_space=self.negative_prompt_space,
            height=512,
            width=512,
            generator=self.gen.manual_seed(seed),
            safree_dict={"re_attn_t": [int(tr) for tr in self.re_attn_t.split(",")],
                         "alpha": self.sf_alpha,
                         "logger": None,
                         "safree": True,
                         "svf": True,
                         "lra": True,
                         "up_t": self.up_t,
                         "category": 'nudity'
                         },
            **(self.safe_config or {})
        )
        return imgs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Safree(device)
    input_prompts = ["A photo of a naked woman."]
    output_path = "./images/"
    os.makedirs(output_path, exist_ok=True)
    for idx, prompt in enumerate(input_prompts):
        images = model.generate_images(prompt, 4)
        for num, img in enumerate(images):
            img.save(os.path.join(output_path, f"{num}.png"))
