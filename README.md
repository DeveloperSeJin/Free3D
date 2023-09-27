# Free3D
기존 text to 3D 인공지능 모델은 input prompt를 고정하여 사용하였다.
본 프로젝트에서는 chatbot을 사용하여 input prompt를 보안하고 input의 부족한 부분을 찾아낸다.

chatbot은 Large Language Model(LLM)인 Flan-T5를 사용하여 구현하였다.
또한 input의 부족한 태그를 찾기 위하여 distillBert를 사용해 NER모델을 구현하였다.

text-to-2D에선 stable diffusion을 사용하였고, 2D-to-3D에선 shape-e를 사용하였다.

# Table of Contents
1. [Samples](#Samples)
2. [Recommended-specifications](#Recommended-specifications)
3. [Usage](#Usage)

# Samples
<img width="80%" src="https://github.com/DeveloperSeJin/Free3D/assets/114290488/58a7e87e-a75f-4ca9-b712-0776eb6c5835">

# Recommended-specifications  
+ **Ubuntu 18.04 with python 3.9 & torch 2.0.1 + CUDA 11.7 on a RTX 2080Ti and NVIDIA T4**
  
+ GPU  
at least 16G of vram(GPU Memory)
+ Python 3  
We tested all process(text processing, image generation, 3d reconstruction) on **Python 3.9**. So we do not guarantee other python version but it may also be working on other python version.
# Usage

# Acknowledgement  
* stable-dreamfusion
```
@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}
```

* DreamFusion Paper
```
@article{poole2022dreamfusion,
    author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
    title = {DreamFusion: Text-to-3D using 2D Diffusion},
    journal = {arXiv},
    year = {2022},
}
```

* Realfusion
```
@inproceedings{melaskyriazi2023realfusion,
  author = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
  title = {RealFusion: 360 Reconstruction of Any Object from a Single Image},
  booktitle={CVPR}
  year = {2023},
  url = {https://arxiv.org/abs/2302.10663},
}
```
