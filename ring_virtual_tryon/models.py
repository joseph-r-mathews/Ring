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
        self.unet = unet
        self.cond = cond          

        # Freeze UNet weights
        for p in self.unet.parameters():
            p.requires_grad = False

        # Extract key modules for explicit access
        self.get_time_embed   = self.unet.get_time_embed
        self.time_proj        = self.unet.time_proj   # may be used by get_time_embed internally
        self.time_embedding   = self.unet.time_embedding
        self.conv_in          = self.unet.conv_in
        self.conv_out         = self.unet.conv_out
        self.down_blocks      = self.unet.down_blocks
        self.up_blocks        = self.unet.up_blocks
        self.mid_block        = self.unet.mid_block
        
        # These are used in the post-processing phase:
        self.conv_norm_out = getattr(self.unet, "conv_norm_out", None)
        self.conv_act      = getattr(self.unet, "conv_act", None)

    def forward(self, sample: torch.Tensor,
                    timestep: torch.Tensor,
                    encoder_hidden_states: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
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

        # --- Process the conditioning features.
        cond_skips = self.cond(c, emb)

        # --- 5. Process through up blocks.
        for up_block in self.up_blocks:
            num_resnets = len(up_block.resnets)
            # Pop residuals from down blocks
            res_hidden_states_tuple = down_block_res_samples[-num_resnets:]
            down_block_res_samples = down_block_res_samples[:-num_resnets]
            # Pop residuals from conditioning features
            cond_slice = cond_skips[-num_resnets:]
            cond_skips = cond_skips[:-num_resnets]
            
            # Create or reuse a list of ZeroConv2d layers to adjust conditioning features.
            if not hasattr(up_block, "cond_conv"):
                up_block.cond_conv = nn.ModuleList([ZeroConv2d(cond.shape[1]) for cond in cond_slice])
            
            # Fuse the skip features: here, we add the processed conditioning features to the image features.
            merged_skips = [
                img + conv(cond)
                for img, cond, conv in zip(res_hidden_states_tuple, cond_slice, up_block.cond_conv)
            ]

            # Sample
            sample = up_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=tuple(merged_skips),
                encoder_hidden_states=encoder_hidden_states
            )

        # --- 6. Post-process: normalization & final convolution.
        if self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample
    

