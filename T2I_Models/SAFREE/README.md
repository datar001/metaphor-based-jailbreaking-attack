# [ICLR 2025] SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation
[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://safree-safe-t2i-t2v.github.io/)  [![arXiv](https://img.shields.io/badge/arXiv-2410.12761-b31b1b.svg)](https://arxiv.org/abs/2410.12761)


### Authors: [Jaehong Yoon*](https://jaehong31.github.io/), [Shoubin Yu*](https://yui010206.github.io/), [Vaidehi Patil](https://vaidehi99.github.io/), [Huaxiu Yao](https://www.huaxiuyao.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
### University of North Carolina at Chapel Hill

<div align=center> 
<img src="./assets/teaser-new.png" alt="teaser image" width="800"/>
</div>

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n SAFREE python=3.10.13
conda activate SAFREE
```


2. build from source

```bash
pip install -r requirements.txt
```


### Safe generation using SD-v1.4 by removing nudity concepts

```bash
bash scripts/run_nudity.sh
```
### Safe generation using SD-XL by removing nudity concepts

```bash
bash scripts/run_nudity_sdxl.sh
```
### COCO image generation using SD-v1.4 while removing nudity concepts

```bash
bash scripts/run_coco.sh
```
### COCO image generation using SD-XL while removing nudity concepts

```bash
bash scripts/run_coco_sdxl.sh
```
### Artist style image generation using SD-v1.4 while removing "Van Gogh" and "Kelly McKernan"

```bash
bash scripts/run_artist.sh
```

# Acknowledgments
The code is built upon [Prompting4Debugging](https://github.com/mbzuai-oryx/Video-LLaVA), [Diffusers](https://github.com/huggingface/diffusers/tree/main), [CogVideoX](https://github.com/THUDM/CogVideo?tab=readme-ov-file), and [ZeroScopeT2V](https://github.com/ExponentialML/ComfyUI_ModelScopeT2V).

# Reference
Please cite our paper if you use our models in your works:

```bibtex
@article{yoon2024safree,
  title={SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation},
  author={Yoon, Jaehong and Yu, Shoubin and Patil, Vaidehi and Yao, Huaxiu and Bansal, Mohit},
  journal={ICLR},
  year={2025}
}
