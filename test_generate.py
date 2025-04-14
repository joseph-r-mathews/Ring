
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import torch
import torch.optim as optim
from tqdm import tqdm
import safetensors.torch # Needed explicitly for diffusers
from ring_virtual_tryon.models import ConditioningEncoder,MainUNet
from ring_virtual_tryon.utils import load_architecture

vae, unetA, unetB = load_architecture()

# Recreate model architecture
model = MainUNet(unetA, ConditioningEncoder(unetB))

print("🔄 Loading model weights from checkpoints/main_unet.pth")
model.load_state_dict(torch.load("checkpoints/main_unet.pth", map_location="cpu"))
print("✅ Model loaded and ready for inference")

model.eval()

# ------------------------------ Testing ----------------------------------
@torch.no_grad()
def generate_hand_with_ring(model, vae, ring_img_path, masked_img_path, num_timesteps):
    """
    Generate a hand image wearing the ring, conditioned on ring and masked-hand images.
    """
    model.eval()

    # ------------ 1. Load & preprocess images -----------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512)),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    ring_img   = transform(Image.open(ring_img_path).convert("RGB")).unsqueeze(0)
    masked_img = transform(Image.open(masked_img_path).convert("RGB")).unsqueeze(0)

    # ------------ 2. Encode conditioning images ----------------
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    cond = (vae.encode(ring_img).latent_dist.mode() + vae.encode(masked_img).latent_dist.mode()) * scaling

    # ------------ 3. Initialize noisy latent -------------------
    xt = torch.randn((1, 4, 64, 64))

    # ------------ 4. Precompute diffusion schedule -------------
    betas = torch.linspace(1e-4, 0.02, num_timesteps)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    # ------------ 5. Reverse diffusion loop --------------------
    for t in tqdm(reversed(range(num_timesteps)), desc="Sampling"):
        t_tensor = torch.tensor([t])
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]

        pred_noise = model(xt, t_tensor, cond)
        print(f"[Step t={t}] pred_noise stats: min={pred_noise.min():.6f}, "
              f"max={pred_noise.max():.6f}, mean={pred_noise.mean():.6f}")
        # DDPM x_{t-1} update
        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        print(f"[Step t={t}] xt stats: min={xt.min():.6f}, max={xt.max():.6f}, mean={xt.mean():.6f}")
        xt = coef1 * (xt - (coef2 * pred_noise))

        if t > 0:
            noise = torch.randn_like(xt)
            xt = xt + beta_t.sqrt() * noise

    x0_latent = xt

    return x0_latent

# ------------------------- View Generated Image ----------------------------

import matplotlib.pyplot as plt
from PIL import ImageFilter

x0_latent = generate_hand_with_ring(
    model=model,
    vae=vae,
    ring_img_path=r"C:\Users\Joe\Desktop\Data\1\ring.jpeg",
    masked_img_path=r"C:\Users\Joe\Desktop\Data\1\masked.jpeg",
    num_timesteps=10
)

#Decoded raw stats: min=-1.0141, max=0.9925, mean=-0.1280

scaling = getattr(vae.config, "scaling_factor", 1.0)
decoded_raw = vae.decode(x0_latent/scaling).sample  # Shape: (1, 3, 512, 512)

# Print raw statistics to verify range
print("Decoded raw stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
    decoded_raw.min().item(), decoded_raw.max().item(), decoded_raw.mean().item()))

# Post-process the decoded tensor:
# 1. Clamp to [-1, 1] (optional if your outputs are already mostly in that range)
decoded_clamped = decoded_raw.clamp(-1, 1)
# 2. Normalize to [0, 1]
decoded_norm = (decoded_clamped + 1) / 2
# 3. Remove the batch dimension and reorder dimensions to (H, W, C)
decoded_img = decoded_norm.squeeze().permute(1, 2, 0).detach().cpu().numpy()
# 4. Convert to 8-bit pixel values
decoded_img = (decoded_img * 255).astype("uint8")

# Create and show the image
from PIL import Image
reconstructed_image = Image.fromarray(decoded_img)
reconstructed_image.show()
