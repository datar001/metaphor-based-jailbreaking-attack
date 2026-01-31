
# SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation


This is the implementation for SAFREE with ZeroScopeT2V model

## Install Dependencies & Download Pre-trained T2V Models
Please refer to this repo installation/preparation [page](https://github.com/ExponentialML/Text-To-Video-Finetuning).

## Download safe T2V test data

We upload our safe T2V meta (selected from [SafeSora](https://github.com/PKU-Alignment/safe-sora) benchmark) data [here](../datasets/) for easy setup.

## Running SAFREE with ZeroScopeT2V

```bash
python test.sh
```

## Evaluate SafeT2V with GPT4
We provide our evaluation scripts for SafeT2V evaluation following prompting engineering in [T2VsafetyBench](https://arxiv.org/abs/2407.05965).

Please see details in jupyter notebook [here](./eval_with_GPT4.ipynb).



# Acknowledgments
The code is built upon [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) project.