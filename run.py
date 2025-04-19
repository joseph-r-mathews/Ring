import torch
import torch.optim as optim
# From other modules
from ring_virtual_tryon.models import MainUNet,ConditioningEncoder
from ring_virtual_tryon.train import train_loop
from ring_virtual_tryon.utils import load_architecture

# --------------------------- Model Initialization -----------------------------
vae, unetA, unetB, tokenizer, text_encoder, scheduler = load_architecture()

model = MainUNet(unetA,ConditioningEncoder(unetB))
loss_fn_diffusion = torch.nn.MSELoss()
loss_fn_img = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------ Training Call ----------------------------------

train_loop(model, vae, tokenizer, text_encoder, scheduler, optimizer, 
           loss_fn_diffusion, loss_fn_img,
           batch_size = 1, shuffle=True, nepochs = 10, device = None, 
           image_file_path=r"C:\Users\Joe\Desktop\Data")

torch.save(model.state_dict(), "checkpoints/main_unet.pth")
print("✅ Model saved to checkpoints/main_unet.pth")