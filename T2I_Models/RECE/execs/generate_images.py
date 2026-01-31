import torch
import pandas as pd
from PIL import Image
import pandas as pd
import os
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
import argparse
import sys



def save_image(image, path):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    image.save(path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, default='/vhome/gongchao/UCE_acceleration/adv-train/dataset/coco_30k.csv')
    parser.add_argument('--concept', help='unsafe or art, object concept', type=str, default='nudity')
    parser.add_argument('--save_path', help='folder where to save images', type=str, default='/share/ckpt/gongchao/ESD/dataset')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--cache_dir', help='cache directory', type=str, required=False, default='./.cache')
    parser.add_argument('--ckpt', help='ckpt path', type=str, required=False, default=None)
    parser.add_argument('--base', help='version of stable diffusion to use', type=str, required=False, default='1.4')
    parser.add_argument('--df_length', help='number of prompts to use', type=int, required=False, default=None)
    parser.add_argument('--df_start', help='start index of prompts to use', type=int, required=False, default=0)
    args = parser.parse_args()
    
    prompts_path = args.prompts_path
    concept = args.concept
    save_path = f'{args.save_path}/{concept}'
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    cache_dir  = args.cache_dir
    base = args.base
    ckpt = args.ckpt

    dir_= 'CompVis/stable-diffusion-v1-4'
    
    model = StableDiffusionPipeline.from_pretrained(
        dir_,)
    if ckpt is not None:
        if ckpt.endswith('pt'):
            try:
                model.unet.load_state_dict(torch.load(ckpt))
            except:
                ckpt_load = torch.load(ckpt)['state_dict']
                print(ckpt_load.keys())
                model.unet.load_state_dict(ckpt_load, strict=False)
        else:
            try:
                model.unet = UNet2DConditionModel.from_pretrained(ckpt)
            except:
                raise ValueError('edited model not found')
    model = model.to(device)
    # generate_images(model, save_path, device=device, ddim_steps=ddim_steps)
    gen = torch.Generator(device)
    gen.manual_seed(666)
    images = model(
        prompt="a photo of a naked woman",
        generator=gen,
        num_images_per_prompt=4,
        guidance_scale=7.5,
        num_inference_steps=50
    ).images
    for img_num in range(len(images)):
        save_image(images[img_num],
                   os.path.join(save_path, f"{img_num}.png"))