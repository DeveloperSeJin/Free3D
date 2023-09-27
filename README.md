# Free3D

## Table of Contents
1. [Samples](#Samples)
2. [Recommended-specifications](#Recommended-specifications)
3. [Usage](#Usage)

기존 text to 3D 인공지능 모델은 input prompt를 고정하여 사용하였다.
본 프로젝트에서는 chatbot을 사용하여 input prompt를 보안하고 input의 부족한 부분을 찾아낸다.

chatbot은 Large Language Model(LLM)인 Flan-T5를 사용하여 구현하였다.
또한 input의 부족한 태그를 찾기 위하여 distillBert를 사용해 NER모델을 구현하였다.

text-to-2D에선 stable diffusion을 사용하였고, 2D-to-3D에선 shape-e를 사용하였다.

## Samples
<img width="60%" src="https://github.com/DeveloperSeJin/Free3D/assets/114290488/58a7e87e-a75f-4ca9-b712-0776eb6c5835">

## Recommended-specifications
vram >= 16G

## Usage
