#from diffusers import DDPMPipeline #options
from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
import torch
import cv2
from PIL import Image
import numpy as np
import random


#모델 연결

def model_run():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16) 
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    return pipe

def model_run2():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16) 
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    return pipe

def model_modify():
    depth_estimator = pipeline('depth-estimation')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    return controlnet, pipe

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def run(prompt):
    ran_seed = random.randint(1, 30) 
    torch.cuda.empty_cache()
    pipe = model_run()
    prompt = prompt + ", best quality, extremely detailed, Uncropped, Sole entity, standalone element"
    generator = torch.Generator(device="cuda").manual_seed(ran_seed)
    image = pipe(
    prompt,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    generator=generator,
    num_inference_steps=20,
    )
    return image[0][0]

def run2(prompt):
    ran_seed = random.randint(1, 30) 
    torch.cuda.empty_cache()
    pipe = model_run2()
    prompt = prompt + ", best quality, extremely detailed, Uncropped, Sole entity, standalone element"
    generator = torch.Generator(device="cuda").manual_seed(ran_seed)
    image = pipe(
    prompt,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    generator=generator,
    num_inference_steps=20,
    )
    return image[0][0]

def modify(prompt):
    torch.cuda.empty_cache()
    depth_estimator = pipeline('depth-estimation')
    #prompt = prompt + ", best quality, extremely detailed"
    image = load_image("./image.png")
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    image = pipe(prompt, image, num_inference_steps=20).images[0]
    return image
