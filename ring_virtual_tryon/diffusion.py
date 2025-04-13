import torch

def make_beta_schedule(num_timesteps,beta_start=1e-4,beta_end=.02):
    """Creates a linear beta schedule for the diffusion process."""
    betas = torch.linspace(beta_start,beta_end,num_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas,dim=0)
    return torch.sqrt(alphas_cumprod),torch.sqrt(1.0 - alphas_cumprod)


def forward_diffusion_sample(x0,t,epsilon,sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod):
    """
    Applies forward diffusion to x0 at timestep t.
    Returns the noisy version xt.
    """
    sqrt_alpha_t = sqrt_alpha_cumprod[t].view(-1,1,1,1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
    return sqrt_alpha_t*x0 + sqrt_one_minus_alpha_t*epsilon

def train_step(model, x0, t, c, optimizer, loss_fn, sqrt_alpha_cumprod, sqrt_one_minus_alphas_cumprod):
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
    epsilon = torch.randn(x0.shape)
    xt = forward_diffusion_sample(x0,t,epsilon,sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod)
    predicted_noise = model(xt,t,c)
    loss = loss_fn(predicted_noise,epsilon)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss