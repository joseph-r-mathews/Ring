
import torch
import torch.optim as optim
# From other modules
from ring_virtual_tryon.models import MainUNet,ConditioningEncoder
from ring_virtual_tryon.train import train_loop
from ring_virtual_tryon.utils import load_architecture

# --------------------------- Model Initialization -----------------------------
vae, unetA, unetB, tokenizer, text_encoder, scheduler = load_architecture()

model = MainUNet(unetA,ConditioningEncoder(unetB))
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------- Settings -----------------------------
device = None
num_inference_steps=5
guidance_scale=7.5
generator=None
height=512
width=512

if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Encode Prompt -----------------------------
prompt = "hand with ring"
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
latent_shape = (1, unetB.config.in_channels, height // 8, width // 8)
latents = torch.randn(latent_shape, generator=generator, device=device)

# --- 4. Set Timesteps ---
scheduler.set_timesteps(num_inference_steps)


# ------------------------------ Training Call ----------------------------------
num_timesteps = 1000        
batch_size = 1             
nepochs = 10        
shuffle = True

#model, x0, t, encoder_hidden_states, c, vae, optimizer, loss_fn, scheduler
train_loop(model,vae,num_timesteps,batch_size,shuffle,nepochs,optimizer,loss_fn)

torch.save(model.state_dict(), "checkpoints/main_unet.pth")
print("✅ Model saved to checkpoints/main_unet.pth")