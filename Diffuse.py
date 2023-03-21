#from diffusers import DDPMPipeline #options  -> DDPM pipeline
from diffusers import StableDiffusionPipeline
import torch
#모델 연결
def run(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)  
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0] 
    image.save('./dsad.jpg')
    return image