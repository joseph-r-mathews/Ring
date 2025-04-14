from .dataset import CachedRingLatents
from torch.utils.data import DataLoader
import torch
from .diffusion import train_step,make_beta_schedule


def train_loop(model, vae, tokenizer, text_encoder, scheduler, optimizer, loss_fn,
               num_inference_steps=1000, batch_size=1, shuffle=True, nepochs=10,
               guidance_scale=7.5, height=512, width=512, generator = None, device = None, 
               image_file_path=r"C:\Users\Joe\Desktop\Data"):
               
    """
    Full training loop for the ring-conditioned diffusion model.

    Args:
        model: MainUNet
        vae: Stable Diffusion VAE for encoding images
        num_timesteps: total diffusion steps (T)
        batch_size: batch size for training
        shuffle: whether to shuffle data loader
        nepochs: number of epochs to train
        optimizer: optimizer instance
        loss_fn: training loss (e.g. MSE)
        image_file_path: root path to training image triplets
    """




    data = DataLoader(CachedRingLatents(image_file_path, vae), batch_size=batch_size, shuffle=shuffle)


    

    sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod = make_beta_schedule(num_inference_steps,beta_start=1e-4,beta_end=.02)
    for epoch in range(nepochs):
        epoch_loss = 0.0
        for batch_idx, (c, x0) in enumerate(data):

            # Sample random diffusion step
            t = torch.randint(0,num_inference_steps,(batch_size,),device=x0.device)


            loss = train_step(model,x0,t,c,vae,optimizer,loss_fn,scheduler)

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{nepochs}], Batch [{batch_idx}/{len(data)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(data)
        print(f"Epoch [{epoch+1}/{nepochs}] Average Loss: {avg_loss:.4f}")






# if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------- Encode Prompt -----------------------------
# prompt = "hand with ring"
# # Tokenize and encode prompt into text embeddings.
# text_inputs = tokenizer(
#     prompt,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     truncation=True,
#     return_tensors="pt",
# )
# text_input_ids = text_inputs.input_ids.to(device)
# text_embeds = text_encoder(text_input_ids).last_hidden_state

# # For classifier free guidance, generate unconditional (empty prompt) embeddings.
# uncond_inputs = tokenizer(
#     [""] * (1 if isinstance(prompt, str) else len(prompt)),
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     truncation=True,
#     return_tensors="pt",
# )
# uncond_ids = uncond_inputs.input_ids.to(device)
# uncond_embeds = text_encoder(uncond_ids).last_hidden_state

# # Concatenate for guidance.
# if guidance_scale > 1.0:
#     # The first half (for unconditional), the second half (for prompt).
#     text_embeds = torch.cat([uncond_embeds, text_embeds])

# # --- 3. Prepare Latent Variables ---
# # Note: The latent tensor spatial dims are usually height/8 x width/8.
# latent_shape = (1, unetB.config.in_channels, height // 8, width // 8)
# latents = torch.randn(latent_shape, generator=generator, device=device)

# # --- 4. Set Timesteps ---
# scheduler.set_timesteps(num_inference_steps)