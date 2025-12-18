import torch
import numpy as np

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError

def get_orthogonal_directions(directions):
    x, y, z = torch.split(directions, 1, dim=-1) # pn,1
    otho0 = torch.cat([y,-x,torch.zeros_like(x)],-1)
    otho1 = torch.cat([-z,torch.zeros_like(x),x],-1)
    mask0 = torch.norm(otho0,dim=-1)>torch.norm(otho1,dim=-1)
    mask1 = ~mask0
    otho = torch.zeros_like(directions)
    otho[mask0] = otho0[mask0]
    otho[mask1] = otho1[mask1]
    otho = torch.nn.functional.normalize(otho, dim=-1)
    return otho

def sample_diffuse_directions(normals, diffuse_direction_samples, is_train):
    # normals [pn,3]
    z = normals # pn,3
    x = get_orthogonal_directions(normals) # pn,3
    y = torch.cross(z, x, dim=-1) # pn,3

    # project onto this tangent space
    az, el = torch.split(diffuse_direction_samples,1,dim=1) # sn,1
    el, az = el.unsqueeze(0), az.unsqueeze(0)
    az = az * np.pi * 2
    el_sqrt = torch.sqrt(el+1e-7)
    if is_train:
        az = (az + torch.rand(z.shape[0], 1, 1, device=az.device) * np.pi * 2) % (2 * np.pi)
    coeff_z = torch.sqrt(1 - el + 1e-7)
    coeff_x = el_sqrt * torch.cos(az)
    coeff_y = el_sqrt * torch.sin(az)

    directions = coeff_x * x.unsqueeze(1) + coeff_y * y.unsqueeze(1) + coeff_z * z.unsqueeze(1) # pn,sn,3
    return directions

