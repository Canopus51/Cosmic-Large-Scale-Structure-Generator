import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from unet import UNet
from utils_diffusion import get_diffusion_params

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
T_STEPS = 400
IMG_SIZE = 256
SAVE_DIR = "checkpoints"
DATA_PATH = "quijote_data.npy"

# Ensure checkpoint directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    # 1. Data Loading and Preprocessing
    try:
        raw_data = np.load(DATA_PATH)
        
        # Ensure data has channel dimension (N, 1, H, W)
        if len(raw_data.shape) == 3:
            raw_data = np.expand_dims(raw_data, axis=1)
        
        # Convert to torch tensor and normalize if not already in [-1, 1]
        train_tensor = torch.from_numpy(raw_data).float()
        
        dataset = TensorDataset(train_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Dataset loaded. Shape: {raw_data.shape}")
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return

    # 2. Model and Diffusion Initialization
    model = UNet(time_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Get diffusion coefficients
    params = get_diffusion_params(T=T_STEPS, device=DEVICE)
    sqrt_alphas_cumprod = torch.sqrt(params["alphas_cumprod"])
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - params["alphas_cumprod"])

    # 3. Training Loop
    model.train()
    print(f"Training started on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for (x0,) in pbar:
            x0 = x0.to(DEVICE)
            current_batch_size = x0.shape[0]
            
            # Sample random timesteps for each image in batch
            t = torch.randint(0, T_STEPS, (current_batch_size,), device=DEVICE).long()
            
            # Generate noise
            noise = torch.randn_like(x0)
            
            # Forward Diffusion Process: q(xt | x0)
            sqrt_ac_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_om_ac_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            xt = sqrt_ac_t * x0 + sqrt_om_ac_t * noise
            
            # Backward Process: Predict the added noise
            predicted_noise = model(xt, t)
            
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} | Avg Loss: {avg_loss:.6f}")

    # Final Model Save
    final_model_path = "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_model_path)
    print(f"Training complete. Model saved as {final_model_path}")

if __name__ == "__main__":
    train()
