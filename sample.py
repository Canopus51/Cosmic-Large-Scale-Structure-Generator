import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import numpy as np
import argparse
from unet import UNet
from utils_diffusion import get_diffusion_params, p_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "best_model.pt"
T_STEPS = 400
IMG_SIZE = 256

def run_sampling(save_gif=False):

    model = UNet(time_dim=128).to(DEVICE)
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model weights from: {CKPT_PATH}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{CKPT_PATH}' not found. Please ensure the file is in the current directory.")
        return


    params = get_diffusion_params(T=T_STEPS, device=DEVICE)
    

    print(f"Generating Large Scale Structure on {DEVICE}...")
    # noise
    x = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE)
    
    frames = []

    for i in reversed(range(T_STEPS)):
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        
        # one step (x_t -> x_{t-1})
        x = p_sample(model, x, t, params)
        
        if save_gif:
            img_frame = x[0, 0].detach().cpu().clamp(-1, 1).numpy()
            img_normalized = (img_frame + 1) / 2.0
            img_color = cm.viridis(img_normalized)
            img_rgb = (img_color[:, :, :3] * 255).astype(np.uint8)
            frames.append(img_rgb)
        
        if i % 50 == 0:
            print(f"Sampling steps remaining: {i}")

    # --- visualize ---
    # save png
    final_img = x[0, 0].detach().cpu().clamp(-1, 1).numpy()
    plt.imsave("ddpm_result.png", final_img, cmap='viridis')
    print("Final image saved: ddpm_result.png")

    # save gif
    if save_gif:
        print(f"Generating GIF (Total: {len(frames)} frames)...")
        imageio.mimsave("ddpm_process.gif", frames, fps=10, loop=0)
        print("Save sample process: ddpm_process.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmic Large Structure Generator")
    # "--gif" parameter
    parser.add_argument("--gif", action="store_true", help="Whether you need sample process GIF")
    args = parser.parse_args()

    run_sampling(save_gif=args.gif)
