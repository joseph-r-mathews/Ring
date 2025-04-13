import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch


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



class ZeroConv2d(nn.Module):
    """
    A 1x1 convolution layer initialized to zero.
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)



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

            if not hasattr(ub, "cond_conv"):
            # Create a ModuleList of ZeroConv2d layers, one per skip connection.
                ub.cond_conv = nn.ModuleList([
                    ZeroConv2d(cond.shape[1]) for cond in cond_slice
                ])
            merged_skips = [
                img + conv(cond)
                for img, cond, conv in zip(img_slice, cond_slice, ub.cond_conv)
            ]

            h = ub(h, tuple(merged_skips), temb=temb)

        return self.conv_out(h)