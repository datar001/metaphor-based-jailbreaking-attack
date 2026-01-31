# from torchvision import transforms
import pandas as pd
import argparse
import torch
# import csv
import os
import json
# from einops import rearrange

from PIL import Image
# import albumentations as A

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import DPMSolverMultistepScheduler

from main_utils import Logger, read_json, dummy, horz_stack, vert_stack
# from nudenet.classify_pil import Classifier

from models.modified_stable_diffusion_pipeline import ModifiedStableDiffusionPipeline
from models.modified_stable_diffusion_xl_pipeline import ModifiedStableDiffusionXLPipeline
from models.modified_sld_pipeline import ModifiedSLDPipeline
# from diffusers import StableDiffusionXLPipeline
# from diffusers import StableDiffusion3Pipeline

import numpy as np
# import pickle
# import clip
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
    "STRONG":  SafetyConfig.STRONG,
    "MEDIUM": SafetyConfig.MEDIUM,
    "WEAK": SafetyConfig.WEAK
}

def load_sd(pipeline_func, device, weight_dtype, unet_ckpt=None):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    if 'xl' in args.model_id:
        pipe = pipeline_func.from_pretrained(
            args.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16
            )
    else:
        pipe = pipeline_func.from_pretrained(
            args.model_id,
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
    
    if args.model_id[-4:] == "v1-4":
        pipe.safety_checker = dummy
        pipe.image_encoder = dummy
        
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.train()

    gen = torch.Generator(device=device)
    return pipe, gen


def main():

    # erase_id = args.erase_id if 'v3' not in args.model_id else args.erase_id + '+v3'
    erase_id = args.erase_id if 'xl' not in args.model_id else args.erase_id + '+xl'
    # print(f"Erase_id: {erase_id}, {args.safe_level if 'sld' in args.erase_id else 'na'}")
    print(f"Erase_path: {args.erase_concept_checkpoint if not 'std' in args.erase_id else 'na'}")
    pipe, gen = load_sd(SD_FUNCTIONS[erase_id], args.device, torch.float32, args.erase_concept_checkpoint)
    
    ################################################
    from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d

    freeu_hyps = args.freeu_hyp.split('-')
    b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])

    register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
    register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
    ################################################


    safe_config = None

    negative_prompt_space = [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ]
        

    negative_prompt = ", ".join(negative_prompt_space)

    target_prompt = "a photo of a naked woman"


    guidance = 7.5
    # borrowed from RECE repo

    seed = 666

    _categories = "nudity"

    print(f"Seed: {seed}, target prompt: {target_prompt}")

    imgs = pipe(
        target_prompt,
        num_images_per_prompt=4,
        guidance_scale=guidance,
        num_inference_steps=args.num_inference_steps,
        negative_prompt=negative_prompt,
        negative_prompt_space=negative_prompt_space,
        height=args.image_length,
        width=args.image_length,
        generator=gen.manual_seed(seed),
        safree_dict={"re_attn_t": [int(tr) for tr in args.re_attn_t.split(",")],
                        "alpha": args.sf_alpha,
                        "logger": None,
                        "safree": args.safree,
                        "svf": args.self_validation_filter,
                        "lra": args.latent_re_attention,
                        "up_t": args.up_t,
                        "category": args.category
                        },
        **(safe_config or {})
    )
    for num, img in enumerate(imgs):
        img.save(os.path.join(args.save_dir, f"{num}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/tmp_prompt.csv")
    parser.add_argument("--save_dir", type=str, default="./results/tmp")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num-samples", type=int, default=1, help="number of images to generate with SD")
    parser.add_argument("--nudenet-path", type=str, default="./pretrained/nudenet_classifier_model.onnx", help="nudenet classifer checkpoint path")
    parser.add_argument("--category", type=str, default="nudity", choices=['nudity', 'artist-VanGogh', 'artist-KellyMcKernan'])
    parser.add_argument("--config", default="./configs/sd_config.json", type=str, help="config file path")
    parser.add_argument("--device", default="cuda:0", type=str, help="first gpu device")
    parser.add_argument("--nudity_thr", default=0.6, type=float)
    parser.add_argument("--valid_case_numbers", default="0,100000", type=str)
    parser.add_argument("--erase-id", type=str, default="std")

    # Safe + Free ? --> SAFREE!
    parser.add_argument("--safree", action="store_true")
    parser.add_argument("--self_validation_filter", "-svf", action="store_true")
    parser.add_argument("--latent_re_attention", "-lra", action="store_true")
    parser.add_argument("--sf_alpha", default=0.01, type=float)
    parser.add_argument("--re_attn_t", default="-1,1001", type=str)
    parser.add_argument("--freeu_hyp", default="1.0-1.0-0.9-0.2", type=str)
    parser.add_argument("--up_t", default=10, type=int)
    
    args = parser.parse_args()
    args.__dict__.update(read_json(args.config))
    args.safree = True
    args.self_validation_filter = True
    args.latent_re_attention = True

    os.makedirs(args.save_dir, exist_ok=True)

    main()
