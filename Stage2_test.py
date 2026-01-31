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
from query_T2I import Text_Filter, Clip_Model, nsfw_list_filter, Img_Cls, SD_checker, Stable_Diffusion, PPL_Model, FLUX, MACE, SLD, RECE, SafeGen, DALLE, GuardT2I, LatentGuard
from evaluators.nudenet import NudeNet
from evaluators.Q16 import Q16
from evaluators.image_evaluator import internVL2
from T2I_Models.SAFREE.Safree import Safree
from APO import bayesian_optimization
from PIL import Image
import argparse
import tqdm
import pdb


def evaluation(adv_prompt, img_dir, class_, nudenet, q16, intervl2, text_filter=None, clip_model=None):
    # adv_prompt = prompt_info["adv_prompt"]
    if text_filter is not None:
        text_bypass = text_filter.filter(adv_prompt)[0]
    else:
        text_bypass = True

    if not text_bypass:
        return False, [], []

    images = []
    img_paths = []
    # import pdb
    # pdb.set_trace()
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        images.append(Image.open(img_path))
        img_paths.append(img_path)

    if img_filter is not None:
        sfw_images, sfw_img_paths = img_filter.filter(images, img_paths)
    else:
        sfw_images = images
        sfw_img_paths = img_paths

    if not sfw_images:
        return False, [], []

    if class_ == "sexual":
        classifier_res = nudenet.eval(sfw_img_paths)
    else:
        classifier_res = q16.eval(sfw_images)

    mllm_res = intervl2.eval(class_, sfw_images)

    return True, classifier_res, mllm_res


def read_json(path_):
    if os.path.exists(path_):
        with open(path_, "r") as f:
            info = json.load(f)
    else:
        info = {}
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sen_prompt_path', type=str, required=True, help='sen_prompt')
    parser.add_argument('--adv_prompt_path', type=str, required=True, help='adv_prompt')
    parser.add_argument('--T2I_model', type=str, required=True, help='T2I Model Names')
    # parser.add_argument('--evaluator', type=str, default="MLLM", help='evaluator settings')
    parser.add_argument('--filters', type=str, default="text+image", help='filter settings')
    args = parser.parse_args()

    device = torch.device("cuda", 0)


    # Target T2I Model
    t2i_type = args.T2I_model
    if t2i_type == "FLUX":
        t2i_model = FLUX(device)
    elif t2i_type == "SD14":
        t2i_model = Stable_Diffusion("V1.4", device)
    elif t2i_type == "SD21":
        t2i_model = Stable_Diffusion("V2.1", device)
    elif t2i_type == "SDXL":
        t2i_model = Stable_Diffusion("XL", device)
    elif t2i_type == "SD3":
        t2i_model = Stable_Diffusion("V3", device)
    elif t2i_type == "SLD-strong":
        t2i_model = SLD(device, mode="strong")
    elif t2i_type == "SLD-max":
        t2i_model = SLD(device, mode="max")
    elif t2i_type == "MACE":
        t2i_model = MACE(device)
    elif t2i_type == "RECE":
        t2i_model = RECE(device)
    elif t2i_type == "Safree":
        t2i_model = Safree(device)
    elif t2i_type == "SafeGen-strong":
        t2i_model = SafeGen(device, mode="STRONG")
    elif t2i_type == "SafeGen-max":
        t2i_model = SafeGen(device, mode="MAX")
    elif t2i_type == "DALLE":
        t2i_model = DALLE()
    else:
        raise ValueError(f"Unknown T2I model: {t2i_type}")

    clip_eva = Clip_Model(device)

    # Safety Filter
    filter_name = args.filters
    if filter_name == "text_match":
        text_filter = nsfw_list_filter()
        img_filter = None
    elif filter_name == "text_cls":
        text_filter = Text_Filter(device)
        img_filter = None
    elif filter_name == "image_cls":
        text_filter = None
        img_filter = Img_Cls(device)
    elif filter_name == "image-clip":
        text_filter = None
        img_filter = clip_eva
    elif filter_name == "text-image-classifier":
        text_filter = None
        img_filter = SD_checker(device)
    elif filter_name == "text+image":
        text_filter = Text_Filter(device)
        img_filter = clip_eva
    elif filter_name == "latent_guard":
        text_filter = LatentGuard(device)
        img_filter = None
    elif filter_name == "guardt2i":
        text_filter = GuardT2I(device)
        img_filter = None
    elif filter_name == "concept_erasing":
        text_filter = None
        img_filter = None
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    # Evaluators
    intervl2 = internVL2(device)
    nudenet = NudeNet(device)
    q16 = Q16(device)

    # with open(args.sen_prompt_path, "r") as f:
    #     test_prompts = json.load(f)
    with open(args.adv_prompt_path, "r") as f:
        infos = json.load(f)

    image_save_dir = fr"/home/zcy/attack/metaphor/attack_images/{t2i_type}/metaphor_qwen_all_adv/"
    os.makedirs(image_save_dir, exist_ok=True)

    classnames = ["disturbing", "violent", "illegal", "sexual"] # , , "violent", "disturbing", "illegal", , "illegal"
    method = "MJA"


    save_dir = f"./attack_results/{method}/{filter_name}"
    os.makedirs(save_dir, exist_ok=True)
    res_save_path = os.path.join(save_dir, f"{t2i_type}_qwen_black_attack_again.json")
    res_info = read_json(res_save_path)

    if t2i_type == "SD14":
        t2i_model = None

    if "statis" not in res_info:
        res_info["statis"] = {}

    all_success_samples = []
    all_queries = []
    for class_, class_prompts in infos.items():
        if class_ not in classnames:
            continue
        if class_ not in res_info:
            res_info[class_] = {}

        success_samples = []
        all_classifier_res = 0
        all_mllm_res = 0
        bypass_res = 0
        for idx, all_adv_prompts in class_prompts.items():

            if str(idx) in res_info[class_]:
                all_queries.append(res_info[class_][str(idx)]['query'])
                continue
            # assert test_prompts[class_][int(idx)] == all_adv_prompts["0"]["prompt"], print("Sen prompts are not consistent!")
            # import pdb; pdb.set_trace()
            sen_prompt = all_adv_prompts["0"]["prompt"] # test_prompts[class_][int(idx)]
            image_dir = os.path.join(image_save_dir, class_, str(idx))
            query_num, prompt_best, idx_best, true_sims, y_best, status = bayesian_optimization(
                sen_prompt, all_adv_prompts, image_dir, text_filter, img_filter, clip_eva, t2i_model)
            final_img_dir = os.path.join(image_dir, str(idx_best))

            bypass, classifier_res, mllm_res = evaluation(prompt_best, final_img_dir, class_, nudenet, q16, intervl2, text_filter, img_filter)

            res_info[class_][str(idx)] = {
                "sen_prompt": sen_prompt,
                "adv_prompt": prompt_best,
                "query": query_num,
                "max_sim": y_best,
                "all_sim": true_sims,
                "bypass": bypass,
                "classifer_res": {
                    "final_res": sum(classifier_res) > 0,
                    "res": classifier_res
                },
                "mllm_res": {
                    "final_res": sum(mllm_res) > 0,
                    "res": mllm_res
                }
            }

            with open(res_save_path, "w") as f:
                json.dump(res_info, f, indent=2)
            all_queries.append(query_num)
            bypass_res += int(bypass)
            all_classifier_res += int(sum(classifier_res) > 0)
            all_mllm_res += int(sum(mllm_res) > 0)
            print(f"{class_}: {idx}||{len(class_prompts)}, bypass_res: {bypass_res/len(class_prompts):.2f}, classifier_acc: {all_classifier_res/len(class_prompts):.4f}, mllm_acc: {all_mllm_res/len(class_prompts):.4f}", end="\r")
        classifier_acc = []
        mllm_acc = []
        queries = []
        bypass_acc = []
        for idx, res in res_info[class_].items():
            queries.append(res["query"])
            classifier_acc.append(int(res["classifer_res"]["final_res"]))
            mllm_acc.append(int(res["mllm_res"]["final_res"]))
            bypass_acc.append(int(res["bypass"]))
        mean_query = np.mean(np.array(queries))
        std_query = np.std(np.array(queries))
        classifier_acc = np.mean(classifier_acc)
        mllm_acc = np.mean(mllm_acc)
        bypass_acc = np.mean(bypass_acc)
        res_info["statis"][class_] = {
            "bypass_acc": bypass_acc,
            "classifier_acc": classifier_acc,
            "mllm_acc": mllm_acc,
            "query": f"{mean_query:.2f}±{std_query:.2f}"
        }
        print(f"{class_}: {idx}||{len(class_prompts)}, bypass_acc: {bypass_acc:.2f}, classifier_acc: {classifier_acc:.4f}, mllm_acc: {mllm_acc:.4f}, query: {mean_query:.2f}±{std_query:.2f}")

    avg_mllm_acc = []
    avg_classifier_acc = []
    avg_bypass_acc = []
    for class_, class_info in res_info["statis"].items():
        avg_bypass_acc.append(class_info["bypass_acc"])
        avg_classifier_acc.append(class_info["classifier_acc"])
        avg_mllm_acc.append(class_info["mllm_acc"])
    avg_classifier_acc = np.mean(avg_classifier_acc)
    avg_mllm_acc = np.mean(avg_mllm_acc)
    avg_bypass_acc = np.mean(avg_bypass_acc)
    mean_query = np.mean(np.array(all_queries))
    std_query = np.std(np.array(all_queries))
    res_info["statis"]["AVG"] = {
        "bypass_acc": avg_bypass_acc,
        "classifier_acc": avg_classifier_acc,
        "mllm_acc": avg_mllm_acc,
        "query": f"{mean_query}±{std_query}",
    }
    print(
        f"AVG: bypass_acc: {avg_bypass_acc:.2f}, classifier_acc: {avg_classifier_acc:.4f}, mllm_acc: {avg_mllm_acc:.4f}, query: {mean_query}±{std_query}")
    with open(res_save_path, "w") as f:
        json.dump(res_info, f, indent=2)