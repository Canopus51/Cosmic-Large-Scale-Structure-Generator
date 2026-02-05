import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    """
    A Light-weight UNet with Time Embedding for Diffusion-based 
    Large Scale Structure (LSS) generation.
    """
    def __init__(self, time_dim=128):
        super().__init__()
        
        self.time_dim = time_dim

        # --- Time Embedding MLP ---
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # --- Encoder (Downsampling) ---
        # Input: (B, 1, 128, 128)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)             
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) 

        # --- Middle ---
        self.mid = nn.Conv2d(128, 128, kernel_size=3, padding=1)           

        # --- Decoder (Upsampling) ---
        self.dec1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)      
        self.dec2 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)       
        self.dec3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)             

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # --- Time Modulation Layers ---
        self.t_enc1 = nn.Linear(time_dim, 32)
        self.t_enc2 = nn.Linear(time_dim, 64)
        self.t_mid  = nn.Linear(time_dim, 128)

        self.act = nn.SiLU()

    def sinusoidal_embedding(self, t, dim):
        """
        Standard sinusoidal embedding as used in DDPM.
        """
        device = t.device
        half_dim = dim // 2
        emb = torch.arange(half_dim, device=device).float()
        emb = torch.exp(-np.log(10000) * emb / (half_dim - 1))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t):
        # 1. Time Embedding
        t_emb = self.sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # 2. Encoder
        # Add time info via broadcasting: (B, C, 1, 1)
        x1 = self.act(self.enc1(x) + self.t_enc1(t_emb)[:, :, None, None])
        x2 = self.act(self.enc2(x1) + self.t_enc2(t_emb)[:, :, None, None])
        x3 = self.act(self.enc3(x2))

        # 3. Middle
        x_mid = self.act(self.mid(x3) + self.t_mid(t_emb)[:, :, None, None])

        # 4. Decoder with Skip Connections
        # 32x32 -> 64x64
        x_up1 = self.upsample(x_mid)
        x_cat1 = torch.cat([x_up1, x2], dim=1)
        x_d1 = self.act(self.dec1(x_cat1))

        # 64x64 -> 128x128
        x_up2 = self.upsample(x_d1)
        x_cat2 = torch.cat([x_up2, x1], dim=1)
        x_d2 = self.act(self.dec2(x_cat2))

        # Final Projection
        out = self.dec3(x_d2)
        return out

if __name__ == "__main__":
    # Quick architecture test
    model = UNet(time_dim=128)
    test_input = torch.randn(1, 1, 128, 128)
    test_time = torch.tensor([10])
    output = model(test_input, test_time)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
