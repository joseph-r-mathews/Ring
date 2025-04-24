import torch

def forward_diffusion_sample(x0, t, epsilon, sqrt_alpha_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Applies the forward diffusion process to a clean latent x0 at timestep t.

    Args:
        x0 (Tensor): Original latent (B, C, H, W)
        t (Tensor): Timesteps (B,)
        epsilon (Tensor): Gaussian noise (B, C, H, W)
        sqrt_alpha_cumprod (Tensor): Precomputed sqrt(alphā_t) values (T,)
        sqrt_one_minus_alphas_cumprod (Tensor): Precomputed sqrt(1 - alphā_t) values (T,)

    Returns:
        Tensor: Noisy latent xt at timestep t
    """
    sqrt_alpha_t = sqrt_alpha_cumprod[t].view(-1,1,1,1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
    return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * epsilon

def x0_prediction(xt, t, predicted_noise, sqrt_alpha_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Predicts the original latent x0 from noisy latent xt and predicted noise.

    Args:
        xt (Tensor): Noisy latent (B, C, H, W)
        t (Tensor): Timesteps (B,)
        predicted_noise (Tensor): Output from the model (B, C, H, W)
        sqrt_alpha_cumprod (Tensor): Precomputed sqrt(alphā_t) values (T,)
        sqrt_one_minus_alphas_cumprod (Tensor): Precomputed sqrt(1 - alphā_t) values (T,)

    Returns:
        Tensor: Reconstructed latent x0
    """
    return (xt - sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * predicted_noise) / \
              sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)


def train_step(model, vae, scheduler,optimizer, 
               loss_fn_diffusion, loss_fn_img,
               x0, t, encoder_hidden_states , condition_features, device):
    """
    Performs a single training step for the DDPM objective, including optional image-space loss.

    Args:
        model: MainUNet – the denoising network
        vae: AutoencoderKL – used to decode latents to pixel space
        scheduler: DDPM scheduler with alphas_cumprod defined
        optimizer: Optimizer instance
        loss_fn_diffusion: Loss function for noise prediction (e.g., MSE)
        loss_fn_img: Loss function for image (e.g., L1)
        x0 (Tensor): Ground truth latent image (B, 4, 64, 64)
        t (Tensor): Diffusion timestep indices (B,)
        encoder_hidden_states (Tensor): Text embeddings (B, N, D)
        condition_features (Tensor): Conditioning latent features (B, 4, 64, 64)
        device: Computation device

    Returns:
        Tensor: Total loss (scalar)
    """
    model.train()
    optimizer.zero_grad()

    # Ensure VAE is frozen; this prevents gradients from flowing into its parameters.
    for p in vae.parameters():
        p.requires_grad = False # <-- Redundant if already frozen elsewhere; not harmful.

    # Precompute square roots of alphā_t and 1 - alphā_t from scheduler.
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    # Sample Gaussian noise and simulate the forward diffusion process.
    epsilon = torch.randn(x0.shape).to(device)

    # Predict noise given the noisy sample, timestep, text embeddings, and conditioning features.
    xt = forward_diffusion_sample(x0, t, epsilon, sqrt_alpha_cumprod, sqrt_one_minus_alphas_cumprod)
    
    # Compute noise prediction loss (DDPM objective).
    predicted_noise = model(xt, t, encoder_hidden_states, condition_features)

    # Compute MSE loss for noise error.
    loss_diffusion = loss_fn_diffusion(predicted_noise, epsilon)

    print("📦 xt:", xt.device)
    print("📦 model params:", next(model.parameters()).device)
    

    # Reconstruct predicted clean latent x0 from xt and predicted noise.
    x0_pred = x0_prediction(xt, t, predicted_noise, sqrt_alpha_cumprod, sqrt_one_minus_alphas_cumprod)
              
    # Decode both predicted and ground-truth latents to pixel space for image loss.
    with torch.no_grad():
        target_img = vae.decode(x0 / vae.config.scaling_factor)["sample"]
    
    decoded_img = vae.decode(x0_pred / vae.config.scaling_factor)["sample"]

    print("📏 predicted_noise:", predicted_noise.device)
    print("📷 decoded_img:", decoded_img.device)
    print("📷 target_img:", target_img.device)

    # Compute L1 loss in pixel space.
    loss_img = loss_fn_img(decoded_img, target_img)

    lambda_img = 0.1 # Tune this weight as needed.

    # Combine both losses: diffusion loss + image reconstruction loss.
    total_loss = loss_diffusion + lambda_img * loss_img

    total_loss.backward()
    optimizer.step()
    return total_loss



# import torch
# import torch.optim as optim
# # From other modules
# from models import MainUNet,ConditioningEncoder
# from dataset import CachedRingLatents
# from torch.utils.data import DataLoader
# from utils import load_architecture
# batch_size = 1
# shuffle=True
# device="cpu"

# vae, unetA, unetB, tokenizer, text_encoder, scheduler = load_architecture()

# model = MainUNet(unetA,ConditioningEncoder(unetB))
# loss_fn_diffusion = torch.nn.MSELoss()
# loss_fn_img = torch.nn.L1Loss()

# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# #"/Users/jm/Downloads/Data"
# image_file_path=r"C:\Users\Joe\Desktop\Data"

# data = DataLoader(CachedRingLatents(image_file_path, vae), batch_size, shuffle)
# # --- Encode Prompt ---
#     # Tokenize and encode prompt into text embeddings.
# prompt = "hand with ring on the ring finger"
# text_inputs = tokenizer(
#     prompt,
#     padding="max_length",
#     max_length=tokenizer.model_max_length,
#     truncation=True,
#     return_tensors="pt",
# )
# text_input_ids = text_inputs.input_ids.to(device)

# with torch.no_grad():
#     text_embeds = text_encoder(text_input_ids).last_hidden_state

# text_embeds = text_embeds.detach()

# condition_features, x0 = next(iter(data))
# t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)

# train_step(
#                 model, vae, scheduler, optimizer, 
#                 loss_fn_diffusion, loss_fn_img,
#                 x0, t, text_embeds , condition_features, device
#             )
