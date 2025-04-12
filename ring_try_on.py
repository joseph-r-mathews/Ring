from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torchvision import transforms
import os 
from diffusers.models import AutoencoderKL
from diffusers import UNet2DConditionModel
import torch
import torch.optim as optim
from tqdm import tqdm
import safetensors.torch # Needed explicitly for diffusers

class RingData(Dataset):
    """
    Dataset for loading triplets of images:
    - 'ring.jpeg': the ring image
    - 'masked.jpeg': the masked hand image
    - 'wearing.jpeg': the ground truth hand wearing the ring
    """
    def __init__(self,root_path):

        self.path = root_path
        self.subfolders = [str(sub_dir.path) for sub_dir in os.scandir(root_path) if sub_dir.is_dir()]
        data_names = ['wearing.jpeg','masked.jpeg','ring.jpeg']

        self.obs = [] 

        for folder in self.subfolders:
            contents = [o for o in os.listdir(folder)] 
    
            if set(data_names).issubset(contents): 
                self.obs.append((os.path.join(folder,"ring.jpeg"),os.path.join(folder,'masked.jpeg'),os.path.join(folder,'wearing.jpeg')))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,512)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # Normalizes pixel values to [-1,1] range.
        ])
        return tuple(transform(Image.open(image).convert("RGB")) for image in self.obs[index])
    
def strip_cross_attention(unet):
    """
    Replaces all cross-attention layers (.attn2) in a Stable Diffusion UNet 
    with identity modules.

    This is useful for replacing text-based conditioning with image-based 
    conditioning. Cross-attention layers are used for text-to-image tasks; 
    when conditioning on images, these layers are unnecessary and are 
    replaced with no-op identity modules to avoid parameter updates.

    Args:
        unet (UNet2DConditionModel): A pretrained Stable Diffusion UNet.

    Returns:
        UNet2DConditionModel: The modified UNet with stripped cross-attention.
    """
    class IdentityAttn(nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
        ):
            return hidden_states

    for m in unet.modules():
        if hasattr(m, "attn2"):
            m.attn2 = IdentityAttn()
    return unet


class ConditioningEncoder(nn.Module):
    """
    Encoder used to extract conditioning features from ring and masked-hand images.
    
    This class reuses the encoder (downsampling) portion of a pretrained Stable Diffusion UNet.
    All cross-attention layers are stripped. The model outputs 12 residual feature maps
    used to guide the upsampling path of the main UNet.
    
    Args:
        unet (UNet2DConditionModel): A pretrained Stable Diffusion UNet instance.
        trainable (bool): Whether the encoder is trainable (default: True).
    """

    def __init__(self, unet: UNet2DConditionModel, trainable=True):
        super().__init__()
        self.unet = strip_cross_attention(unet) 
        
        self.conv_in     = self.unet.conv_in
        self.down_blocks = self.unet.down_blocks

    def forward(self, x, temb):
        """
        Args:
            x (Tensor): Conditioning latent (B, 4, 64, 64)
            temb (Tensor): Time embedding (B, 1280)

        Returns:
            Tuple[Tensor]: A tuple of 12 feature maps from the downsampling path.

        """
        h = self.conv_in(x) 
        skips = (h,) # Include initial feature map for compatibility with 12-skip design  
        for db in self.down_blocks:
            h, res = db(h, temb)  
            skips += res
        
        assert(len(skips) == 12) # Expected 12 skip connections for decoder compatibility.

        return skips

class MainUNet(nn.Module):
    """
    Diffusion UNet modified for ring-conditioned image synthesis.

    Uses a pretrained Stable Diffusion UNet as the backbone with parameters frozen.
    A second encoder processes the conditioning image inputs. The decoder merges
    skip connections from both the input path and the conditioning encoder.

    Args:
        unet (UNet2DConditionModel): Pretrained diffusion UNet with frozen weights.
        cond (ConditioningEncoder): A trainable encoder for conditioning features.
    """
    def __init__(self, unet: UNet2DConditionModel, cond: ConditioningEncoder):
        super().__init__()
        self.unet  = strip_cross_attention(unet)
        self.cond  = cond          

        # Freeze UNet weights
        for p in self.unet.parameters():
            p.requires_grad = False

        # Extract key modules for explicit access
        self.time_proj      = self.unet.time_proj
        self.time_embedding = self.unet.time_embedding
        self.conv_in        = self.unet.conv_in
        self.conv_out       = self.unet.conv_out
        self.down_blocks    = self.unet.down_blocks
        self.up_blocks      = self.unet.up_blocks
        self.mid_block      = self.unet.mid_block

    def forward(self, x, t, c):
        """
        Args:
            x (Tensor): Noisy input latent (B, 4, 64, 64)
            t (Tensor): Diffusion timestep (B,)
            c (Tensor): Conditioning latent (B, 4, 64, 64)

        Returns:
            Tensor: Predicted noise tensor (B, 4, 64, 64)
        """
       
        # Embed time using MLP
        temb = self.time_embedding(self.time_proj(t))

        # Encode input latent and collect skip connections
        h = self.conv_in(x)
        img_skips = (h,)
        for db in self.down_blocks:
            h, res = db(h, temb)
            img_skips += res
        assert(len(img_skips) == 12)

        # Mid block processing
        h = self.mid_block(h, temb)
        
        # Encode conditioning image
        cond_skips = self.cond(c, temb) 

        # Decoder with per-layer conditioning fusion
        for ub in self.up_blocks:
            
            n = len(ub.resnets)           
            img_slice  = img_skips[-n:];  img_skips  = img_skips[:-n]
            cond_slice = cond_skips[-n:]; cond_skips = cond_skips[:-n]

            # Learnable scalar per fusion layer
            if not hasattr(ub, "cond_weights"):
                ub.cond_weights = nn.Parameter(torch.zeros(n))
            merged_skips = [
                img + w * cond
                for img, cond, w in zip(img_slice, cond_slice, ub.cond_weights)
            ]

            h = ub(h, tuple(merged_skips), temb=temb)

        return self.conv_out(h)

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


class CachedRingLatents(Dataset):
    """
    Dataset that encodes all images using a pretrained VAE and caches the latent tensors.

    Stores:
    - c = latent(ring + masked)
    - x0 = latent(ground truth)
    """

    def __init__(self, root_path, vae):
        super().__init__()
        self.vae = vae.eval()
        self.scaling = getattr(vae.config, "scaling_factor", 1.0)

        self.obs = RingData(root_path)
        self.latents = []

        with torch.no_grad():
            for i in range(len(self.obs)):
                ring, masked, wearing = self.obs[i]

                ring_lat    = self.vae.encode(ring.unsqueeze(0)).latent_dist.mode() * self.scaling
                masked_lat  = self.vae.encode(masked.unsqueeze(0)).latent_dist.mode() * self.scaling
                wearing_lat = self.vae.encode(wearing.unsqueeze(0)).latent_dist.mode() * self.scaling

                cond = ring_lat + masked_lat
                self.latents.append((cond.squeeze(0), wearing_lat.squeeze(0)))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]  # returns (cond_latent, gt_latent)


def train_loop(model,vae,num_timesteps,batch_size,shuffle,nepochs,optimizer,loss_fn,image_file_path="/Users/jm/Downloads/Data"):
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
    #scaling = getattr(vae.config, "scaling_factor", 1.0)
    for epoch in range(nepochs):
        epoch_loss = 0.0
        for batch_idx, (c, x0) in enumerate(data):
            # # Encode ring and masked hand as conditioning latent
            # c = (vae.encode(batch[0]).latent_dist.sample() + 
            #      vae.encode(batch[1]).latent_dist.sample()) * scaling

            # # Encode target image
            # x0 = vae.encode(batch[2]).latent_dist.sample() * scaling

            # Sample random diffusion step
            t = torch.randint(0,num_timesteps,(batch_size,),device=x0.device)

            loss = train_step(model,x0,t,c,optimizer,loss_fn,
                              sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod)

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{nepochs}], Batch [{batch_idx}/{len(data)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(data)
        print(f"Epoch [{epoch+1}/{nepochs}] Average Loss: {avg_loss:.4f}")


# --------------------------- Model Initialization -----------------------------
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

model = MainUNet(unetA,ConditioningEncoder(unetB))
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------ Training Call ----------------------------------
num_timesteps = 1000        
batch_size = 1             
nepochs = 2        
shuffle = True

train_loop(model,vae,num_timesteps,batch_size,shuffle,nepochs,optimizer,loss_fn)





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

        # DDPM x_{t-1} update
        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        xt = coef1 * (xt - coef2 * pred_noise)

        if t > 0:
            noise = torch.randn_like(xt)
            xt = xt + beta_t.sqrt() * noise

    x0_latent = xt

    # ------------ 6. Decode to image ----------------------------
    decoded = vae.decode(x0_latent / scaling).sample  # (1, 3, 512, 512)
    decoded = (decoded.clamp(-1, 1) + 1) / 2  # [0,1]
    decoded = decoded.cpu().squeeze().permute(1,2,0).numpy()
    decoded = (decoded * 255).astype("uint8")
    image = Image.fromarray(decoded)

    return image

# ------------------------- View Generated Image ----------------------------

import matplotlib.pyplot as plt
from PIL import ImageFilter

image = generate_hand_with_ring(
    model=model,
    vae=vae,
    ring_img_path="/path/to/ring.jpeg",
    masked_img_path="/path/to/masked.jpeg",
    num_timesteps=1000,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Assuming you have a PIL image `image`
sharp_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
sharp_image.show()




# ------------------------- Print Learnable Parameters ----------------------------

# print("Trainable parameters:")
# for name, p in model.named_parameters():
#     if p.requires_grad:
#         print(f"  {name}  |  shape: {p.shape}")


# ------------------------------ Debugging ----------------------------------
# sqrt_alpha_cumprod,sqrt_one_minus_alphas_cumprod =  make_beta_schedule(10,beta_start=1e-4,beta_end=.02)
# x0 = torch.randn([10,4,64,64])
# t = torch.randint(0,10,(10,))
# epsilon = torch.randn(x0.shape)


# t = torch.randint(0,10,(1,))
# temb = unetB.time_embedding(unetB.time_proj(t))
# cond = ConditioningEncoder(unetA)
# c = torch.randn([1,4,64,64])
# x = torch.randn([1,4,64,64])
# model = MainUNet(unetB,cond)

# out=model(x,t,c)