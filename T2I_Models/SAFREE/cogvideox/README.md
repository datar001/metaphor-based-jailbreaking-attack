# SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation

This is the implementation for SAFREE with CogVideoX

## Install Dependencies & Download Pre-trained T2V Models
Please refer to CogVideo repo installation/preparation [page](https://github.com/THUDM/CogVideo/blob/main/sat/README.md)

## Download safe T2V test data

We upload our safe T2V meta (selected from [SafeSora](https://github.com/PKU-Alignment/safe-sora) benchmark) data [here](../datasets/) for easy setup.

## Running SAFREE with CogVideoX

```bash
python cli_demo.py --prompt [Your prompt] --unsafe_concept [user-defined unsafe concept] --model_path [pretrained T2V model path] --output_path ./output.mp4
```

## Evaluate SafeT2V with GPT4
We provide our evaluation scripts for SafeT2V evaluation following prompting engineering in [T2VsafetyBench](https://arxiv.org/abs/2407.05965).

Please see details in jupyter notebook [here](./eval_with_GPT4.ipynb).


## Acknowledgments
The code is built upon [CogVideo](https://github.com/THUDM/CogVideo) project.