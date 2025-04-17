from .dataset import CachedRingLatents
from torch.utils.data import DataLoader
import torch
from .diffusion import train_step

# importlib.reload(diffusion)

def train_loop(model, vae, tokenizer, text_encoder, scheduler, optimizer, 
               loss_fn_diffusion, loss_fn_img,
               batch_size=1, shuffle=True, nepochs=10, device = None, 
               image_file_path=r"C:\Users\Joe\Desktop\Data"):   
    """
    Full training loop for the ring-conditioned diffusion model.

    Args:
        model (MainUNet): The modified UNet model for diffusion.
        vae (AutoencoderKL): Pretrained Stable Diffusion VAE used for encoding/decoding.
        tokenizer (CLIPTokenizer): Tokenizer for encoding text prompts.
        text_encoder (CLIPTextModel): Text encoder (e.g., CLIP) to produce prompt embeddings.
        scheduler (Scheduler): Diffusion scheduler used for alpha schedule and denoising.
        optimizer (Optimizer): Optimizer instance (e.g., Adam).
        loss_fn_diffusion (Loss): Loss function for denoising objective (e.g., MSE).
        loss_fn_img (Loss): Image reconstruction loss (e.g., L1).
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset.
        nepochs (int): Number of training epochs.
        device (torch.device): Device to run training on (GPU or CPU).
        image_file_path (str): Root directory containing image triplets.
    """
    # --- Initialize DataLoader ---
    data = DataLoader(CachedRingLatents(image_file_path, vae), batch_size, shuffle)

    # --- Setup Device ---
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # --- Encode Prompt (shared across all batches) ---
    prompt = "hand with ring on the ring finger"
    
    # Tokenize and encode prompt into CLIP embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        text_embeds = text_encoder(text_input_ids).last_hidden_state
    
    text_embeds = text_embeds.detach() # (1, 77, 768)

    for epoch in range(nepochs):
        epoch_loss = 0.0

        for batch_idx, (condition_features, x0) in enumerate(data):
            # Ground truth target (hand with ring)
            x0 = x0.to(device)
            condition_features = condition_features.to(device) # Conditioning input (ring + masked hand)

            # Sample random diffusion timestep for each sample
            t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)

            # Perform a single training step
            loss = train_step(
                model, vae, scheduler, optimizer, 
                loss_fn_diffusion, loss_fn_img,
                x0, t, text_embeds , condition_features, device
            )

            epoch_loss += loss.item()

            # Print batch-level loss every 10 steps
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{nepochs}], Batch [{batch_idx}/{len(data)}], Loss: {loss.item():.4f}")

        # Compute and print epoch-level average loss
        avg_loss = epoch_loss / len(data)
        print(f"Epoch [{epoch+1}/{nepochs}] Average Loss: {avg_loss:.4f}")