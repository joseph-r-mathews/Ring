import torch.nn as nn
from diffusers import UNet2DConditionModel
import torch


def strip_cross_attention(unet):
    """
    Replaces all cross-attention layers (.attn2) in a Stable Diffusion UNet 
    with identity modules.

    Useful when replacing text-based cross-attention with image-based conditioning.
    Prevents unnecessary computation and avoids updating unused parameters.

    Args:
        unet (UNet2DConditionModel): A pretrained UNet from Stable Diffusion.

    Returns:
        UNet2DConditionModel: Modified UNet with stripped cross-attention layers.
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
    A 1x1 convolution initialized to zero weights and bias.

    Used to fuse conditioning features into UNet skip connections.
    Starts with no influence on the main path but learns during training.
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
    Conditioning encoder that extracts multi-scale features from the input 
    (e.g., ring + masked hand).

    Reuses the downsampling path of a pretrained Stable Diffusion UNet.
    Cross-attention layers are stripped to prevent use of text embeddings.

    Args:
        unet (UNet2DConditionModel): Pretrained UNet used as encoder backbone.
        trainable (bool): Whether encoder weights are trainable (default: True).
    """
    def __init__(self, unet: UNet2DConditionModel, trainable=True):
        super().__init__()
        self.unet = strip_cross_attention(unet) 
        
        self.conv_in     = self.unet.conv_in
        self.down_blocks = self.unet.down_blocks

    def forward(self, x, temb):
        """
        Args:
            x (Tensor): Input conditioning latent (B, 4, 64, 64).
            temb (Tensor): Time embedding (B, 1280).

        Returns:
            Tuple[Tensor]: 12 skip connections from the downsampling path.
        """
        h = self.conv_in(x) 
        skips = (h,)
        for db in self.down_blocks:
            h, res = db(h, temb)  
            skips += res

        return skips
    

class MainUNet(nn.Module):
    """
    UNet for ring-conditioned image synthesis.

    Extends a pretrained Stable Diffusion UNet to include conditioning from 
    a secondary encoder (e.g., ring + masked hand). All original UNet weights 
    are frozen. The decoder path fuses conditioning features at each resolution.

    Args:
        unet (UNet2DConditionModel): Frozen Stable Diffusion UNet.
        cond (ConditioningEncoder): Trainable encoder for conditioning features.
    """
    def __init__(self, unet: UNet2DConditionModel, cond: ConditioningEncoder):
        super().__init__()
        self.unet = unet
        self.cond = cond          

        # Freeze pretrained UNet weights
        for p in self.unet.parameters():
            p.requires_grad = False

        # Extract components for explicit access
        self.get_time_embed   = self.unet.get_time_embed
        self.time_proj        = self.unet.time_proj 
        self.time_embedding   = self.unet.time_embedding
        self.conv_in          = self.unet.conv_in
        self.conv_out         = self.unet.conv_out
        self.down_blocks      = self.unet.down_blocks
        self.up_blocks        = self.unet.up_blocks
        self.mid_block        = self.unet.mid_block
        
        # Optional normalization and activation after final block
        self.conv_norm_out = getattr(self.unet, "conv_norm_out", None)
        self.conv_act      = getattr(self.unet, "conv_act", None)

    def forward(self, sample: torch.Tensor,
                    timestep: torch.Tensor,
                    encoder_hidden_states: torch.Tensor,
                    condition_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model with conditioning injection.

        Args:
            sample (Tensor): Noisy latent input (B, 4, 64, 64).
            timestep (Tensor): Diffusion timestep (B,).
            encoder_hidden_states (Tensor): Text embeddings for cross-attn (B, 77, 768).
            condition_features (Tensor): Conditioning latent (B, 4, 64, 64).

        Returns:
            Tensor: Predicted noise tensor (B, 4, 64, 64).
        """
        # --- Time embedding ---
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        # --- Pre-process input using conv_in and store initial residual ---
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)

        # --- UNet encoder (downsampling path) ---
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states
            )
            down_block_res_samples += res_samples

        # --- mid block (bottleneck) ---
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # --- conditioning features ---
        cond_skips = self.cond(condition_features, emb)

        # --- UNet decoder (upsampling path with conditioning fusion) ---
        for up_block in self.up_blocks:

            # num_resnets usually 2 or 3 in standard Stable Diffusion model
            num_resnets = len(up_block.resnets)

            # Get skip connections from both paths
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

            # Apply upsampling block
            sample = up_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=tuple(merged_skips),
                encoder_hidden_states=encoder_hidden_states
            )

        # --- Post-process: normalization & final convolution ---
        if self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        return self.conv_out(sample)
    

