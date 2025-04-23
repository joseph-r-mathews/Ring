import torch
import torch.optim as optim
# From other modules
from ring_virtual_tryon.models import MainUNet,ConditioningEncoder
from ring_virtual_tryon.train import train_loop
from ring_virtual_tryon.utils import load_architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --------------------------- Model Initialization -----------------------------
vae, unetA, unetB, tokenizer, text_encoder, scheduler = load_architecture()

# Move components to the selected device
vae = vae.to(device)
unetA = unetA.to(device)
unetB = unetB.to(device)
text_encoder = text_encoder.to(device)

model = MainUNet(unetA,ConditioningEncoder(unetB)).to(device)
loss_fn_diffusion = torch.nn.MSELoss()
loss_fn_img = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------ Training Call ----------------------------------

train_loop(model, vae, tokenizer, text_encoder, scheduler, optimizer, 
           loss_fn_diffusion, loss_fn_img,
           batch_size = 1, shuffle=True, nepochs = 10, device = device, 
           image_file_path=r"C:\Users\Joe\Desktop\Data")

torch.save(model.state_dict(), "checkpoints/main_unet.pth")
print("✅ Model saved to checkpoints/main_unet.pth")