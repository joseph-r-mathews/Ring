# python3 -m venv env

# .\env\Scripts\activate



import torch
import torchvision
import torch.nn as nn
from diffusers import UNet2DConditionModel


unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  # or v1-5
    subfolder="unet"
)

# Create dummy inputs.
batch_size = 1
channels = unet.config.in_channels  # typically 4 for Stable Diffusion.
height, width = 64, 64  # latent spatial dimensions
x = torch.randn(batch_size, channels, height, width)

# Dummy timestep; typically one per batch element.
timestep = torch.tensor([10], dtype=torch.long)

# Dummy encoder hidden states.
# For Stable Diffusion, the text encoder outputs are typically [batch, 77, cross_attention_dim].
encoder_hidden_states = torch.randn(batch_size, 77, unet.config.cross_attention_dim)

# Create the time embedding used in the forward pass.
t_emb = unet.time_embedding(unet.time_proj(timestep))

# --- Initial convolution ---
hs = []
h = unet.conv_in(x)
print("After conv_in, shape:", h.shape)
hs.append(h)

# --- Process through down blocks ---
for idx, down_block in enumerate(unet.down_blocks):
    h, res_samples = down_block(h, t_emb, encoder_hidden_states)
    print(f"After down block {idx}, shape: {h.shape}")
    for j, res in enumerate(res_samples):
        print(f"   Down block {idx} - residual {j} shape: {res.shape}")
    hs.extend(res_samples)

h = unet.mid_block(h, t_emb, encoder_hidden_states)
print("After mid block, shape:", h.shape)

first_up_block = unet.up_blocks[0]
if hasattr(first_up_block, "resnets"):
    expected_num = len(first_up_block.resnets)
    print("First up block expects", expected_num, "residual connections (skip features).")
else:
    print("First up block does not have an attribute 'resnets'.")
    expected_num = 0

if expected_num > 0:
    # The up block will use the last 'expected_num' skip connections from the collected list.
    expected_skips = hs[-expected_num:]
    for i, skip in enumerate(expected_skips):
        print(f"Expected skip connection {i} should have shape: {skip.shape}")

# Now, pop the expected number of skip connections from the list.
skip_connections = []
for i in range(expected_num):
    skip = hs.pop()  # Pop from the end (most recent added).
    print(f"Popped skip connection {i} with shape:", skip.shape)
    skip_connections.append(skip)

# Reverse the order (if needed) so that the ordering matches the network's expectations.
skip_connections = tuple(skip_connections[::-1])

# Run the first up block.
h = first_up_block(h, skip_connections, t_emb)
print("After first up block, shape:", h.shape)



####################################
# Second up block (CrossAttnUpBlock2D)
####################################
# Get the second up block from the UNet
second_up_block = unet.up_blocks[1]
if hasattr(second_up_block, "resnets"):
    expected_num_second = len(second_up_block.resnets)
    print("\nSecond up block expects", expected_num_second, "skip connections.")
else:
    expected_num_second = 0
    print("\nSecond up block does not have a 'resnets' attribute.")

print("Remaining skip connections in hs before second up block:", len(hs))
if expected_num_second > 0:
    # The expected skip connections for the second up block are the last expected_num_second from hs.
    expected_skips_second = hs[-expected_num_second:]
    for i, skip in enumerate(expected_skips_second):
        print(f"Expected skip connection {i} for second up block should have shape: {skip.shape}")

# Pop the skip connections for the second up block.
skip_connections_second = []
for i in range(expected_num_second):
    skip = hs.pop()  # Pop from the end.
    print(f"Popped skip connection {i} for second up block with shape:", skip.shape)
    skip_connections_second.append(skip)
# Reverse order to match the expected order in the block.
skip_connections_second = tuple(skip_connections_second[::-1])

# Now call the second up block.
# Note: For CrossAttnUpBlock2D, the order of arguments is:
#   hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, ...
h = second_up_block(h, skip_connections_second, t_emb, encoder_hidden_states)
print("\nAfter second up block, shape:", h.shape)


# WHEN CHANNEL NUMBER or spatial dimension NOT THE SAME YOU GET
# RuntimeError: Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [1920] and input of shape [1, 2560, 16, 16]



####################################
# Third up block (assumed CrossAttnUpBlock2D)
####################################
third_up_block = unet.up_blocks[2]
if hasattr(third_up_block, "resnets"):
    expected_num_third = len(third_up_block.resnets)
    print("\nThird up block expects", expected_num_third, "skip connections.")
else:
    expected_num_third = 0
    print("\nThird up block does not have a 'resnets' attribute.")

print("Remaining skip connections in hs before third up block:", len(hs))
if expected_num_third > 0:
    expected_skips_third = hs[-expected_num_third:]
    for i, skip in enumerate(expected_skips_third):
        print(f"Expected skip connection {i} for third up block should have shape: {skip.shape}")

skip_connections_third = []
for i in range(expected_num_third):
    skip = hs.pop()
    print(f"Popped skip connection {i} for third up block with shape:", skip.shape)
    skip_connections_third.append(skip)
skip_connections_third = tuple(skip_connections_third[::-1])

h = third_up_block(h, skip_connections_third, t_emb, encoder_hidden_states)
print("\nAfter third up block, shape:", h.shape)



# --- Final up block (fourth block) ---
final_up_block = unet.up_blocks[-1]
# Determine the expected number of residual skip connections.
if hasattr(final_up_block, "resnets"):
    expected_num_final = len(final_up_block.resnets)
    print("\nFinal up block expects", expected_num_final, "skip connections (residuals).")
else:
    expected_num_final = 0
    print("\nFinal up block does not have a 'resnets' attribute.")

print("Remaining skip connections in hs before final up block:", len(hs))
if expected_num_final > 0:
    expected_skips_final = hs[-expected_num_final:]
    for i, skip in enumerate(expected_skips_final):
        print(f"Expected skip connection {i} for final up block should have shape: {skip.shape}")

# Pop the expected number of skip connections for the final up block.
skip_connections_final = []
for i in range(expected_num_final):
    skip = hs.pop()  # Pop from the end.
    print(f"Popped skip connection {i} for final up block with shape:", skip.shape)
    skip_connections_final.append(skip)

# Reverse the ordering so that the earliest (lowest-resolution) skip is first.
skip_connections_final = tuple(skip_connections_final[::-1])

# Call the final up block.
# Note: CrossAttnUpBlock2D.forward expects: 
#   (hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, ...)
h = final_up_block(h, skip_connections_final, t_emb, encoder_hidden_states)
print("\nAfter final up block, shape:", h.shape)






