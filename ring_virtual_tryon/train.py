from .dataset import CachedRingLatents
from torch.utils.data import DataLoader
import torch
from .diffusion import train_step,make_beta_schedule


def train_loop(model,vae,num_timesteps,batch_size,shuffle,nepochs,optimizer,loss_fn,
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

    data = DataLoader(CachedRingLatents(image_file_path, vae),
                      batch_size=batch_size, shuffle=shuffle)
    sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod = make_beta_schedule(num_timesteps,beta_start=1e-4,beta_end=.02)
    for epoch in range(nepochs):
        epoch_loss = 0.0
        for batch_idx, (c, x0) in enumerate(data):

            # Sample random diffusion step
            t = torch.randint(0,num_timesteps,(batch_size,),device=x0.device)

            loss = train_step(model,x0,t,c,vae,optimizer,loss_fn,
                              sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod)

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{nepochs}], Batch [{batch_idx}/{len(data)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(data)
        print(f"Epoch [{epoch+1}/{nepochs}] Average Loss: {avg_loss:.4f}")