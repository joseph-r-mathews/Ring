import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

def simple_generate(
    prompt,
    height=512,
    width=512,
    num_inference_steps=5,
    guidance_scale=7.5,
    generator=None,
    device=None
):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # --- 1. Load Models & Tokenizer ---
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    ).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    # --- 2. Encode Prompt ---
    # Tokenize and encode prompt into text embeddings.
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    text_embeds = text_encoder(text_input_ids).last_hidden_state

    # For classifier free guidance, generate unconditional (empty prompt) embeddings.
    uncond_inputs = tokenizer(
        [""] * (1 if isinstance(prompt, str) else len(prompt)),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)
    uncond_embeds = text_encoder(uncond_ids).last_hidden_state

    # Concatenate for guidance.
    if guidance_scale > 1.0:
        # The first half (for unconditional), the second half (for prompt).
        text_embeds = torch.cat([uncond_embeds, text_embeds])

    # --- 3. Prepare Latent Variables ---
    # Note: The latent tensor spatial dims are usually height/8 x width/8.
    latent_shape = (1, unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device)

    # --- 4. Set Timesteps ---
    scheduler.set_timesteps(num_inference_steps)
    # --- 5. Denoising Loop ---
    for i, t in enumerate(scheduler.timesteps):
        print(f"Step {i+1}/{num_inference_steps}, timestep: {t}")

        # If doing guidance, duplicate latents.
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        
        # Scale latent input as required by the scheduler.
        latent_input = scheduler.scale_model_input(latent_input, t)
        
        # Get noise prediction from the UNet conditioned on the text embeddings.
        noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeds)["sample"]
        
        # Apply classifier free guidance.
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute the previous denoised sample.
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    # --- 6. Decode Latents to Image ---
    # The scaling factor (0.18215) is as used in the original implementation.
    latents = latents / 0.18215
    with torch.no_grad():
        decoded = vae.decode(latents)["sample"]

    # --- 7. Postprocess Image ---
    # Convert images from [-1, 1] to [0, 1] and move channel dim.
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()  # (batch, height, width, channels)
    
    return image

# Example usage:
if __name__ == "__main__":
    prompt = "A beautiful landscape with mountains"
    image = simple_generate(prompt)
    # Here you could convert the numpy array to a PIL image and save or display it.


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Assume `image` is the NumPy array returned from simple_generate
# Convert the first image in the batch from [0, 1] to [0, 255] as uint8
pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

# Option 1: Show using PIL (this opens the default image viewer)
pil_image.show()

# Option 2: Display inline using matplotlib (especially useful in Jupyter notebooks)
plt.imshow(pil_image)
plt.axis("off")
plt.show()




import torch
import types
from diffusers import UNet2DConditionModel

# Set manual seed for reproducibility
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the UNet model from the pretrained checkpoint.
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(device)
unet.eval()  # ensure model is in evaluation mode

# 2. Define a minimal forward pass that should reproduce the default forward.
def forward_minimal(self, sample: torch.Tensor,
                    timestep: torch.Tensor,
                    encoder_hidden_states: torch.Tensor) -> torch.Tensor:
    """
    A minimal forward pass for UNet2DConditionModel based on default settings.
    This implementation reproduces the same output as the full UNet forward.
    """
    # --- 1. Time embedding
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb)

    # --- 2. Pre-process input using conv_in and store initial residual.
    sample = self.conv_in(sample)
    down_block_res_samples = (sample,)

    # --- 3. Process through down blocks.
    for down_block in self.down_blocks:
        sample, res_samples = down_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states
        )
        down_block_res_samples += res_samples

    # --- 4. Process the mid block (bottleneck).
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

    # --- 5. Process through up blocks.
    for up_block in self.up_blocks:
        num_resnets = len(up_block.resnets)
        res_hidden_states_tuple = down_block_res_samples[-num_resnets:]
        down_block_res_samples = down_block_res_samples[:-num_resnets]
        sample = up_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_hidden_states_tuple,
            encoder_hidden_states=encoder_hidden_states
        )

    # --- 6. Post-process: normalization & final convolution.
    if self.conv_norm_out is not None:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

# Bind the minimal forward as a method on our unet instance.
unet.forward_minimal = types.MethodType(forward_minimal, unet)

# 3. Create dummy inputs.
batch_size = 1
in_channels = unet.config.in_channels  # should be 4 per config
height = unet.config.sample_size         # default 64
width = unet.config.sample_size          # default 64

# Create a random latent sample: shape (batch, 4, 64, 64)
sample_input = torch.randn(batch_size, in_channels, height, width, device=device)

# Create a dummy timestep (an integer or tensor) – here we use an integer.
timestep = 10

# Create dummy encoder hidden states.
# Typical CLIP text encoders produce embeddings with ~77 tokens and 768 features.
sequence_length = 77
cross_attention_dim = unet.config.cross_attention_dim  # should be 768 per config
encoder_hidden_states = torch.randn(batch_size, sequence_length, cross_attention_dim, device=device)

# 4. Run the full forward pass.
with torch.no_grad():
    output_full = unet(sample_input, timestep, encoder_hidden_states)
    # The full forward returns a UNet2DConditionOutput (a dataclass) with the attribute "sample"
    output_full_tensor = output_full.sample if hasattr(output_full, 'sample') else output_full

# Run the minimal forward pass.
with torch.no_grad():
    output_minimal_tensor = unet.forward_minimal(sample_input, timestep, encoder_hidden_states)

# 5. Compare the outputs.
print("Output shape (full forward):   ", output_full_tensor.shape)
print("Output shape (minimal forward):", output_minimal_tensor.shape)

# Check if the outputs are equal (within a tolerance).
outputs_equal = torch.allclose(output_full_tensor, output_minimal_tensor, atol=1e-5)
print("Outputs are equal:", outputs_equal)