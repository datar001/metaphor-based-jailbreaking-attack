import json
import os.path
import random
import re
from typing import Any, Iterator, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from query_T2I import Text_Filter, Clip_Model, Stable_Diffusion, PPL_Model, FLUX, MACE, SLD, RECE
from T2I_Models.SAFREE.Safree import Safree
from image_evaluator import internVL2
import argparse
import tqdm
import pdb


def transfer_attack(adv_prompt, t2i_model, image_evaluator, class_, text_filter=None, clip_model=None):
    if text_filter is not None:
        text_bypass_flage = text_filter.reward([adv_prompt])[0]
        if not text_bypass_flage:
            return False, None, None

    images = t2i_model.generate_images(adv_prompt, 4, 666)

    if clip_model is not None:
        safe_images = clip_model.image_filter(images)
        if not safe_images:
            return False, None, None
    else:
        safe_images = images

    if safe_images:
        all_attack_res = image_evaluator.eval(class_, safe_images)
        if sum(all_attack_res) > 0:
            return True, safe_images, all_attack_res
    return False, safe_images, all_attack_res


def read_json(path_):
    if os.path.exists(path_):
        with open(path_, "r") as f:
            info = json.load(f)
    else:
        info = {}
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_prompt_path', type=str, required=True, help='adv_prompt')
    parser.add_argument('--T2I_model', type=str, required=True, help='T2I Model Names')
    args = parser.parse_args()

    device = torch.device("cuda", 0)

    t2i_type = args.T2I_model
    if t2i_type == "FLUX":
        t2i_model = FLUX(device)
    elif t2i_type == "SD14":
        t2i_model = Stable_Diffusion("V1.4", device)
    elif t2i_type == "SDXL":
        t2i_model = Stable_Diffusion("XL", device)
    elif t2i_type == "SD3":
        t2i_model = Stable_Diffusion("V3", device)
    elif t2i_type == "SLD":
        t2i_model = SLD(device)
    elif t2i_type == "MACE":
        t2i_model = MACE(device)
    elif t2i_type == "RECE":
        t2i_model = RECE(device)
    elif t2i_type == "Safree":
        t2i_model = Safree(device)
    elif t2i_type == "DALLE":
        t2i_model = DALLE()
    else:
        raise ValueError(f"Unknown T2I model: {t2i_type}")

    image_evaluator = internVL2(device)
    if t2i_type not in ["FLUX", "SD14", "SD3", "SDXL"]:
        text_filter = None
        clip_model = None
    else:
        text_filter = Text_Filter(device)
        clip_model = Clip_Model(device)

    with open(args.adv_prompt_path, "r") as f:
        infos = json.load(f)

    classnames = ["sexual"] # , "violent", "disturbing", "illegal"
    methods = ["PGJ"]

    save_dir = "./transfer_attack"

    res_save_path = os.path.join(save_dir, f"{t2i_type}_transfer_attack.json")
    res_info = read_json(res_save_path)


    for method, method_data in infos.items():
        if method not in methods:
            continue
        if method not in res_info:
            res_info[method] = {}
        if "statis" not in res_info[method]:
            res_info[method]["statis"] = {}

        all_success_samples = []
        for class_, class_prompts in method_data.items():
            if class_ not in classnames:
                continue
            if class_ not in res_info[method]:
                res_info[method][class_] = {}
            image_save_dir = os.path.join(save_dir, t2i_type, method, class_)
            os.makedirs(image_save_dir, exist_ok=True)

            success_samples = []
            for idx, adv_prompt in enumerate(class_prompts):
                # adv_prompt = idx_info["adv_prompt"]
                final_attack_res, safe_images, all_attack_res = transfer_attack(adv_prompt, t2i_model, image_evaluator, class_, text_filter, clip_model)
                success_samples.append(final_attack_res)
                all_success_samples.append(final_attack_res)
                image_dir = os.path.join(image_save_dir, str(idx))
                os.makedirs(image_dir, exist_ok=True)

                if safe_images:
                    for image_idx in range(len(safe_images)):
                        image_path = os.path.join(image_dir,
                                                  str(image_idx) + "_" + str(all_attack_res[image_idx]) + ".png")
                        safe_images[image_idx].save(image_path)
                res_info[method][class_][str(idx)] = {
                    "success": final_attack_res,
                    "adv_prompt": adv_prompt,
                }
                with open(res_save_path, "w") as f:
                    json.dump(res_info, f, indent=2)

            asr = sum(success_samples) / len(success_samples)
            print(f"{method}--{class_}: asr--{asr:.4f}")
            res_info[method]["statis"][class_] = {
                "ASR": asr
            }
            with open(res_save_path, "w") as f:
                json.dump(res_info, f, indent=2)
        asr = sum(all_success_samples) / len(all_success_samples)
        print(f"AVG: {method}--asr--{asr:.4f}")
        res_info[method]["statis"]["AVG"] = {
            "ASR": asr
        }
        with open(res_save_path, "w") as f:
            json.dump(res_info, f, indent=2)