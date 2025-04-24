from PIL import Image
import os 
from torch.utils.data import Dataset
from torchvision import transforms
import torch


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

                assert next(self.vae.parameters()).device == ring.device, \
                    f"VAE is on {next(self.vae.parameters()).device}, but input is on {ring.device}"

                ring    = ring.to(self.vae.device)
                masked  = masked.to(self.vae.device)
                wearing = wearing.to(self.vae.device)

                ring_lat    = self.vae.encode(ring.unsqueeze(0)).latent_dist.mode() * self.scaling
                masked_lat  = self.vae.encode(masked.unsqueeze(0)).latent_dist.mode() * self.scaling
                wearing_lat = self.vae.encode(wearing.unsqueeze(0)).latent_dist.mode() * self.scaling

                assert ring_lat.device == masked_lat.device == wearing_lat.device, \
                    "Latents are on mismatched devices!"
                
                cond = ring_lat + masked_lat
                self.latents.append((cond.squeeze(0), wearing_lat.squeeze(0)))
                assert self.latents[-1][0].device == next(self.vae.parameters()).device, \
                    "Cached latent is on different device than VAE"



    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx] 