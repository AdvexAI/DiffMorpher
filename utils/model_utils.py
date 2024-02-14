import torch
import torch.nn.functional as F
from torchvision import transforms

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()

    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    if len(size) == 3:
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    else:
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_img(img, resolution=512):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img = transform(img)
    return img.unsqueeze(0)

@torch.no_grad()
def slerp(p0, p1, fract_mixing: float, adain=True):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    p0 = p0.double()
    p1 = p1.double()

    if adain:
        mean1, std1 = calc_mean_std(p0)
        mean2, std2 = calc_mean_std(p1)
        mean = mean1 * (1 - fract_mixing) + mean2 * fract_mixing
        std = std1 * (1 - fract_mixing) + std2 * fract_mixing
        
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1

    if adain:
        interp = F.instance_norm(interp) * std + mean

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp

def modify_t(samples, beta=10):
    
    def mod(t, beta=10):
        return torch.exp(-beta * (t - 0.5)**2)
    
    num_samples = len(samples)
    
    # Calculate the weights for each sample
    weights = mod(samples, beta)
    
    # Normalize the weights to sum to 1
    normalized_weights = weights / weights.sum()
    
    # Sample new indices based on the weights
    indices = torch.multinomial(normalized_weights, num_samples, replacement=True)
    
    # Sort the selected indices to maintain the ascending order
    sorted_indices = torch.sort(indices).values
    
    # Select samples based on the sampled indices
    dense_samples = samples[sorted_indices]
    
    # Add in slight noise to avoid duplicates
    dense_samples += torch.rand_like(dense_samples) * 1e-1
    
    return dense_samples

def do_replace_attn(key: str):
    # return key.startswith('up_blocks.2') or key.startswith('up_blocks.3')
    return key.startswith('up')
