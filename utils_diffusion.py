import torch

def get_diffusion_params(T=400, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Culculate the diffusion parameters"""
    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "T": T
    }

@torch.no_grad()
def p_sample(model, x_t, t, params):
    """DDPM single-step reverse sampling: x_t -> x_{t-1}"""
    eps_theta = model(x_t, t)
    
    beta_t = params["betas"][t].view(-1, 1, 1, 1)
    sqrt_recip_alpha_t = params["sqrt_recip_alphas"][t].view(-1, 1, 1, 1)
    sqrt_om_ac_t = params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)

    # DDPM reverse mean formula
    mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_om_ac_t) * eps_theta)

    # Do not add noise when t=0
    if (t == 0).all():
        return mean
    
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(beta_t) * noise
