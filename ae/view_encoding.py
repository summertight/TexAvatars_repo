import torch
import numpy as np
from einops import rearrange, repeat

def custom_meshgrid(*args):
     # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
     return torch.meshgrid(*args, indexing='ij')
 
def plucker(K, c2w, H, W, device, flip_flag=None):

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  

    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)

    zs = torch.ones_like(i) 
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  
    directions = directions / directions.norm(dim=-1, keepdim=True) 

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  
    rays_o = c2w[..., :3, 3] 
    rays_o = rays_o[:, :, None].expand_as(rays_d) 

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  

    plucker = rearrange(plucker, "b f h w c -> b c f h w")  
    return plucker

def plucker_init():
    return torch.zeros((1, 6, 1, 802, 550))


def to_cam_matrix(cam):
    cam_matrix = {}
    R = torch.tensor(cam.R, dtype=torch.float32).cuda()
    T = torch.tensor(cam.T, dtype=torch.float32).cuda()
    K = torch.tensor([[cam.fl_x, 0, cam.cx],
                    [0, cam.fl_y, cam.cy],
                    [0, 0, 1]], dtype=torch.float32).cuda()
    
    cam_matrix = {
        'R': R.unsqueeze(0).unsqueeze(0), 
        't': T.unsqueeze(0).unsqueeze(0), 
        'K': K.unsqueeze(0)  
    }
    return cam_matrix

def cam_matrix_init():

    b = 1
    cam_matrix = {'R': torch.tensor([[[  # Define a rigid rotation
    [-1/3, -(1/3)**.5, (1/3)**.5],
    [1/3, -(1/3)**.5, -(1/3)**.5],
    [-2/3, 0, -(1/3)**.5]],
    ]]).cuda(), 
    't': torch.tensor([[[2, 2, 2]]]).cuda(), 
    'K':torch.tensor([[
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    ]]).cuda(), 
    }
    return cam_matrix
