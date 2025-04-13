from diffusers.models import AutoencoderKL, UNet2DConditionModel
import safetensors.torch # Needed explicitly for diffusers
import torch

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
    return vae, unetA, unetB