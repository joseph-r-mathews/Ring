import safetensors.torch # Needed explicitly for diffusers
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

def load_architecture():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    unetA = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision="fp16",
        torch_dtype=torch.float32
    )
    unetB = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision="fp16",
        torch_dtype=torch.float32
    )
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    return vae, unetA, unetB, tokenizer, text_encoder, scheduler