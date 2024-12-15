# Contrastive CFG: Improving CFG in Diffusion Models by Contrasting Positive and Negative Concepts

This repository is the official implementation of *[Contrastive CFG: Improving CFG in Diffusion Models by Contrasting Positive and Negative Concepts](https://arxiv.org/abs/2411.17077)* by:

*[Jinho Chang](https://jinhojsk515.github.io/)*, *[Hyungjin Chung](https://www.hj-chung.com/)*, and *[Jong Chul Ye](https://bispl.weebly.com/professor.html)*

![main figure](assets/fig1.png)

[![arXiv](https://img.shields.io/badge/arXiv-2411.17077-b31b1b.svg)](https://arxiv.org/abs/2411.17077)

---
## 1. Abstract

As Classifier-Free Guidance (CFG) has proven effective in conditional diffusion model sampling for improved condition alignment, many applications use a negated CFG term to filter out unwanted features from samples. 
However, simply negating CFG guidance creates an inverted probability distribution, often distorting samples away from the marginal distribution. 

Inspired by recent advances in conditional diffusion models for inverse problems, here we present a novel method to enhance negative CFG guidance using contrastive loss. 
Specifically, our guidance term aligns or repels the denoising direction based on the given condition through contrastive loss, achieving a nearly identical guiding direction to traditional CFG for positive guidance while overcoming the limitations of existing negative guidance methods. 
Experimental results demonstrate that our approach effectively removes undesirable concepts while maintaining sample quality across diverse scenarios, from simple class conditions to complex and overlapping text prompts.

## 2. Conda env setup
Run `pip install -r requirements.txt` to install the required packages into your wanted conda environment.

## 3. Running the code

### Text-to-Image generation with positive & negative prompts

- CFG
```
python -m examples.text_to_img_np --pos_prompt "a photo of a flower." --neg_prompt "a yellow flower." --method "ddim_np_naive" --cfg_guidance 7.5 --n_sample 1 --minibatch 1
```

- ContrastiveCFG (ours)

```
python -m examples.text_to_img_np --pos_prompt "a photo of a flower." --neg_prompt "a yellow flower." --method "ddim_np_ccfg" --cfg_guidance 7.5 --n_sample 1 --minibatch 1
```


## 4. Citation
If you find our method useful, please cite as below or leave a star to this repository.

```
@article{chang2024contrastive,
  title={Contrastive CFG: Improving CFG in Diffusion Models by Contrasting Positive and Negative Concepts},
  author={Chang, Jinho and Chung, Hyungjin and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2411.17077},
  year={2024}
}
```

> [!note] This work is currently in the preprint stage, and there may be some changes to the code.
