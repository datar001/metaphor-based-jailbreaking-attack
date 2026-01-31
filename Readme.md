# Metaphor-based jailbreak attacks on T2I models

If you are interested in our work, please star ⭐ this repository.

#### |Paper [[arXiv]](https://arxiv.org/abs/2512.10766) |

## Abstract

Text-to-image~(T2I) models commonly incorporate defense mechanisms to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attacks have shown that adversarial prompts can effectively bypass these mechanisms and induce T2I models to produce sensitive content, revealing critical safety vulnerabilities. However, existing attack methods implicitly assume that the attacker knows the type of deployed defenses, which limits their effectiveness against unknown or diverse defense mechanisms. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to effectively and efficiently attack diverse defense mechanisms without prior knowledge of their type by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Extensive experiments on T2I models with various external and internal defense mechanisms demonstrate that MJA outperforms six baseline methods, achieving stronger attack performance while using fewer queries.

![Taxonomy](.\figures\visualization.png)

## Setup

```json
Core python dependences:
- pytorch == 2.6.0
- torchvision == 0.21.0
- tokenizers == 0.21.4
- diffusers == 0.32.2
- transformers == 4.50.0
```



## Using MJA to attack T2I models

First, we generate candidate adversarial prompts using MLAG module:

```python
python Stagle1_LMAG.py --exp_save_dir 'stage1_adv_prompts' --sensitive_prompt_path 'test_prompts.json' --metaphor_num 7 --context_num 7
# This step will generate 49~(7*7) adversarial prompts for APO module.
```

Second, we conduct the attack experiments on different T2I models.

```python
python Stage2_test.py --adv_prompt_path 'stage1_adv_prompts/adversarial_prompts.json' --T2I_model 'SD14' --filters 'text+image'
# Noted that, we support various attack settings, including five T2I models, seven internal defense mechanisms, and eight external defense mechanisms
# Common Attack Setting:
# T2I models + External Defense:
#	--T2I models 'any of SD14/SD21/SDXL/SD3/FLUX' 
# 	--filters 'any of text_match/text_cls/image_cls/image-clip/text-image-classifier/text+image/latent_guard/guardt2i'
# T2I models + Internal Defense:
#	--T2I models 'any of SLD-strong/SLD-max/MACE/RECE/Safree/SafeGen-strong/SafeGen-max' 
# 	--filters 'concept_erasing'
# Commercial T2I models
# 	--T2I models 'DALLE3' 
# 	--filters 'concept_erasing'
```



## Resources

Some defense mechanisms rely on the pre-trained models, we list them on the following links:



## Citation

```
@article{zhang2025metaphor,
  title={Metaphor-based jailbreaking attacks on text-to-image models},
  author={Zhang, Chenyu and Ma, Yiwen and Wang, Lanjun and Li, Wenhui and Tu, Yi and Liu, An-An},
  journal={arXiv preprint arXiv:2512.10766},
  year={2025}
}
```

