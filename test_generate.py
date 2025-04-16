from PIL import Image
import torch.nn as nn
from torchvision import transforms
import torch
import torch.optim as optim
from tqdm import tqdm
#import safetensors.torch  # Needed explicitly for diffusers
from ring_virtual_tryon.models import ConditioningEncoder, MainUNet
from ring_virtual_tryon.utils import load_architecture

# Load pretrained components: VAE, two UNets, tokenizer, text encoder, and scheduler.
vae, unetA, unetB, tokenizer, text_encoder, scheduler = load_architecture()

# Recreate the model architecture: using unetA as the main frozen UNet backbone
# and a ConditioningEncoder constructed from unetB.
model = MainUNet(unetA, ConditioningEncoder(unetB))

print("🔄 Loading model weights from checkpoints/main_unet.pth")
model.load_state_dict(torch.load("checkpoints/main_unet.pth", map_location="cpu"))
print("✅ Model loaded and ready for inference")

model.eval()

# ------------------------------ Testing ----------------------------------
@torch.no_grad()

def generate_hand_with_ring(
    model, vae, tokenizer, text_encoder, scheduler,
    ring_img_path, masked_img_path, num_timesteps,
    height=512, width=512, guidance_scale=7.5, 
    generator=None, device=None
):
    """
    Generate a hand image wearing the ring, conditioned on a ring image and a masked-hand image.

    Args:
        model (MainUNet): The diffusion model that predicts noise.
        vae (AutoencoderKL): The VAE for decoding latents into images.
        tokenizer (CLIPTokenizer): Tokenizer for the prompt.
        text_encoder (CLIPTextModel): Pretrained text encoder to process the prompt.
        scheduler: The diffusion scheduler that provides the noise schedule.
        ring_img_path (str): Path to the ring image.
        masked_img_path (str): Path to the masked-hand image.
        num_timesteps (int): Number of timesteps for reverse diffusion.
        height (int): Output image height.
        width (int): Output image width.
        guidance_scale (float): Scale for classifier-free guidance.
        generator: Optional torch.Generator.
        device: The torch device to run computations on.

    Returns:
        numpy.ndarray: The generated image in [0, 1] range with shape (batch, height, width, channels).
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # ---------------- Load & Preprocess Conditioning Images ----------------

    # Define preprocessing transform:
    # - Convert image to tensor.
    # - Resize image to 512x512.
    # - Normalize pixel values to [-1, 1].

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    # Load and transform the ring and masked-hand images; add a batch dimension.
    ring_img   = transform(Image.open(ring_img_path).convert("RGB")).unsqueeze(0)
    masked_img = transform(Image.open(masked_img_path).convert("RGB")).unsqueeze(0)

    # ---------------- Encode Conditioning Images -------------------

    # Get the VAE scaling factor (used in Stable Diffusion)
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    # Encode both images with the VAE encoder; then sum the latent representations.
    # This combination forms the conditioning features.
    condition_features = (vae.encode(ring_img).latent_dist.mode() + 
                          vae.encode(masked_img).latent_dist.mode()) * scaling


    # ---------------- Encode Text Prompt -------------------

    prompt = "hand with ring on the ring finger"

    # Tokenize and encode the prompt with CLIP's text encoder.
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    text_embeds = text_encoder(text_input_ids).last_hidden_state

    # For classifier-free guidance, also encode an unconditional (empty) prompt.
    uncond_inputs = tokenizer(
        [""] * (1 if isinstance(prompt, str) else len(prompt)),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_embeds = text_encoder(uncond_ids).last_hidden_state

    # If guidance is enabled, duplicate the text embeddings by concatenating
    # unconditional and conditional embeddings. Also, duplicate condition_features.
    if guidance_scale > 1.0:
        # After concatenation, text_embeds has a batch size of 2 (first half unconditional, second conditional).
        text_embeds = torch.cat([uncond_embeds, text_embeds])
        # Duplicate the conditioning features to match the duplicated text embeddings.
        condition_features = torch.cat([condition_features] * 2)

    # ---------------- Prepare Initial Latent Variables ----------------

    # Latents are sampled from a standard Gaussian. The spatial dimensions are typically 1/8 of the image.
    latent_shape = (1, unetA.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device)

    # ---------------- Set Scheduler Timesteps ------------------

    scheduler.set_timesteps(num_timesteps)

    # ---------------- Reverse Diffusion (Denoising) Loop ----------------

    for i, t in enumerate(scheduler.timesteps):
        print(f"Step {i+1}/{num_timesteps}, timestep: {t}")

        # When using guidance, duplicate the latents to match the duplicated text embeddings and conditioning.
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        
        # Scale the latent input appropriate for the current timestep.
        latent_input = scheduler.scale_model_input(latent_input, t)

        # Pass through the MainUNet to predict noise.
        noise_pred = model(latent_input, t, text_embeds, condition_features)
        
        # If guidance is enabled, split the noise predictions and combine them using the guidance scale.
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Use the scheduler to step back one diffusion iteration.
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    # ---------------- Decode the Final Latent to an Image ----------------

    # The scaling factor (0.18215) is as used in the original implementation.
    latents = latents / scaling
    with torch.no_grad():
        decoded = vae.decode(latents)["sample"]

    # ---------------- Postprocess the Image ----------------

    # Convert the image from [-1, 1] to [0, 1] and rearrange the dimensions to (batch, H, W, C).
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()  # (batch, height, width, channels)
    
    return image


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image = generate_hand_with_ring(
    model, vae, tokenizer, text_encoder, scheduler,
    "/Users/jm/Downloads/Data/1/ring.jpeg", 
    "/Users/jm/Downloads/Data/1/masked.jpeg", 
    5,
    height=512, width=512, guidance_scale=7.5, 
    generator=None, device=None)

# Assume `image` is the NumPy array returned from simple_generate
# Convert the first image in the batch from [0, 1] to [0, 255] as uint8
pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

# Option 1: Show using PIL (this opens the default image viewer)
pil_image.show()

# Option 2: Display inline using matplotlib (especially useful in Jupyter notebooks)
plt.imshow(pil_image)
plt.axis("off")
plt.show()




#     # ------------ 3. Initialize noisy latent -------------------
#     xt = torch.randn((1, 4, 64, 64))

#     # ------------ 4. Precompute diffusion schedule -------------
#     betas = torch.linspace(1e-4, 0.02, num_timesteps)
#     alphas = 1.0 - betas
#     alpha_cumprod = torch.cumprod(alphas, dim=0)

#     # ------------ 5. Reverse diffusion loop --------------------
#     for t in tqdm(reversed(range(num_timesteps)), desc="Sampling"):
#         t_tensor = torch.tensor([t])
#         beta_t = betas[t]
#         alpha_t = alphas[t]
#         alpha_bar_t = alpha_cumprod[t]

#         pred_noise = model(xt, t_tensor, cond)
#         print(f"[Step t={t}] pred_noise stats: min={pred_noise.min():.6f}, "
#               f"max={pred_noise.max():.6f}, mean={pred_noise.mean():.6f}")
#         # DDPM x_{t-1} update
#         coef1 = 1 / alpha_t.sqrt()
#         coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
#         print(f"[Step t={t}] xt stats: min={xt.min():.6f}, max={xt.max():.6f}, mean={xt.mean():.6f}")
#         xt = coef1 * (xt - (coef2 * pred_noise))

#         if t > 0:
#             noise = torch.randn_like(xt)
#             xt = xt + beta_t.sqrt() * noise

#     x0_latent = xt

#     return x0_latent

# # ------------------------- View Generated Image ----------------------------

# import matplotlib.pyplot as plt
# from PIL import ImageFilter

# x0_latent = generate_hand_with_ring(
#     model=model,
#     vae=vae,
#     ring_img_path=r"C:\Users\Joe\Desktop\Data\1\ring.jpeg",
#     masked_img_path=r"C:\Users\Joe\Desktop\Data\1\masked.jpeg",
#     num_timesteps=10
# )

# #Decoded raw stats: min=-1.0141, max=0.9925, mean=-0.1280

# scaling = getattr(vae.config, "scaling_factor", 1.0)
# decoded_raw = vae.decode(x0_latent/scaling).sample  # Shape: (1, 3, 512, 512)

# # Print raw statistics to verify range
# print("Decoded raw stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
#     decoded_raw.min().item(), decoded_raw.max().item(), decoded_raw.mean().item()))

# # Post-process the decoded tensor:
# # 1. Clamp to [-1, 1] (optional if your outputs are already mostly in that range)
# decoded_clamped = decoded_raw.clamp(-1, 1)
# # 2. Normalize to [0, 1]
# decoded_norm = (decoded_clamped + 1) / 2
# # 3. Remove the batch dimension and reorder dimensions to (H, W, C)
# decoded_img = decoded_norm.squeeze().permute(1, 2, 0).detach().cpu().numpy()
# # 4. Convert to 8-bit pixel values
# decoded_img = (decoded_img * 255).astype("uint8")

# # Create and show the image
# from PIL import Image
# reconstructed_image = Image.fromarray(decoded_img)
# reconstructed_image.show()
