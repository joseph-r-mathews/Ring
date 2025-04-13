
import torch
import torch.optim as optim
# From other modules
from ring_virtual_tryon.models import MainUNet,ConditioningEncoder
from ring_virtual_tryon.train import train_loop
from ring_virtual_tryon.utils import load_architecture

# --------------------------- Model Initialization -----------------------------
vae, unetA, unetB = load_architecture()

model = MainUNet(unetA,ConditioningEncoder(unetB))
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------ Training Call ----------------------------------
num_timesteps = 1000        
batch_size = 1             
nepochs = 10        
shuffle = True

train_loop(model,vae,num_timesteps,batch_size,shuffle,nepochs,optimizer,loss_fn)

torch.save(model.state_dict(), "checkpoints/main_unet.pth")
print("✅ Model saved to checkpoints/main_unet.pth")