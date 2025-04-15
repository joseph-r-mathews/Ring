import torch

# def make_beta_schedule(num_timesteps,beta_start=1e-4,beta_end=.02):
#     """Creates a linear beta schedule for the diffusion process."""
#     betas = torch.linspace(beta_start,beta_end,num_timesteps)
#     alphas_cumprod = torch.cumprod(1.0 - betas,dim=0)
#     return torch.sqrt(alphas_cumprod),torch.sqrt(1.0 - alphas_cumprod)


def forward_diffusion_sample(x0,t,epsilon,sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod):
    """
    Applies forward diffusion to x0 at timestep t.
    Returns the noisy version xt.
    """
    sqrt_alpha_t = sqrt_alpha_cumprod[t].view(-1,1,1,1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
    return sqrt_alpha_t*x0 + sqrt_one_minus_alpha_t*epsilon

def x0_prediction(xt,t,predicted_noise,sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod):
    return (xt - sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * predicted_noise) / \
              sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)


def train_step(model, vae, scheduler, loss_fn, x0, t, encoder_hidden_states ,c, optimizer, device):
    """
    Single training step for DDPM loss.

    Args:
        model: MainUNet
        x0: ground truth latent (B, 4, 64, 64)
        t: diffusion step (B,)
        c: conditioning latent (B, 4, 64, 64)
        optimizer: optimizer instance
        loss_fn: loss function (e.g., MSE)
        sqrt_alpha_cumprod: sqrt(alpha_bar_t)
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t)

    Returns:
        Scalar loss
    """
    model.train()
    optimizer.zero_grad()

    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    epsilon = torch.randn(x0.shape).to(device)

    xt = forward_diffusion_sample(x0,t,epsilon,
                                  sqrt_alpha_cumprod,
                                  sqrt_one_minus_alphas_cumprod)
    
    predicted_noise = model(xt,t,encoder_hidden_states,c)
    
    loss_diffusion = loss_fn(predicted_noise,epsilon)

    # --- Image Reconstruction Loss ---
    x0_pred = x0_prediction(xt,t,predicted_noise,sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod)
              
    with torch.no_grad():
        decoded_img = vae.decode(x0_pred / vae.config.scaling_factor)["sample"]
        target_img = vae.decode(x0 / vae.config.scaling_factor)["sample"]

    loss_img = torch.nn.functional.l1_loss(decoded_img, target_img)

    lambda_img = 0.1 
    total_loss = loss_diffusion + lambda_img * loss_img

    
    total_loss.backward()
    optimizer.step()
    return total_loss