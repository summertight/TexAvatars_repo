# https://github.com/markomih/SplatFields/blob/main/extract_geo.py

import torch
import numpy as np
import os

def query_nn(pts, n_neighbors=5, eps=1e-5):
    from pytorch3d.ops.knn import knn_points
    dists, nn_ix, _ = knn_points(pts.unsqueeze(0), pts.unsqueeze(0), K=n_neighbors, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)
    corss_dists = torch.cdist(pts[nn_ix], pts[nn_ix]) # B x N x N
    # convert distances to spatial weights
    weights = torch.full_like(corss_dists, fill_value=eps)
    weights[corss_dists > eps] = 1.0 / corss_dists[corss_dists > eps]
    weights = weights / weights.sum(-1).sum(-1)[:, None, None].clamp_min(1e-5)
    return weights, nn_ix

def morans_measure(weight, feature):
    """
    Args:
        weight: B x N x N matrix of weights
        feature: B x N x F matrix of features
    Returns:
        moran: Moran's I measure
    """
    # skip the first one which corresponds to the centroid
    assert not torch.isnan(weight).any()
    N = feature.shape[1]
    W = weight.sum(-1).sum(-1)[:, None, None] # B x 1 x 1
    w_ij = (N/W)*weight # B x N x N
    assert not torch.isnan(w_ij).any()
    
    # feature_mean = feature.mean(dim=1, keepdim=True) # B x 1 x F
    # x_mean = feature - feature_mean # B x N x F
    x_mean = feature
    denom = (x_mean**2).sum(dim=1) # B x F

    x_mean_bf = x_mean.contiguous().permute(0, 2, 1).reshape(-1, N) # B*F x N
    x_corr = (x_mean_bf.unsqueeze(-1) @ x_mean_bf.unsqueeze(-2)) # B*F x N x N
    x_corr = x_corr.view(x_mean.shape[0], x_mean.shape[2], N, N) # B x F x N x N

    nom = (w_ij.unsqueeze(1) * x_corr).sum(-1).sum(-1) # B x F
    moran = nom / (denom + 1e-4) # B x F

    return moran.mean()

def morans_loss(weight, feature):
    moran_score = morans_measure(weight, feature)
    moran_loss = 1.0 - moran_score.clamp(0, 1)
    return moran_loss

def neg_morans_loss(weight, feature):
    moran_score = morans_measure(weight, feature)
    moran_loss = moran_score.clamp(0, 1)
    return moran_loss


import torch
import torch.nn.functional as F


def compute_curvature(vertices, faces):
    """
    Compute curvature for each vertex in the mesh.
    
    Args:
        vertices (Tensor): B x V x 3 tensor of mesh vertex positions.
        faces (Tensor): F x 3 tensor of mesh face indices.

    Returns:
        curvature (Tensor): B x V tensor of curvature values.
    """
    # Compute face normals
    v0, v1, v2 = vertices[:, faces[:, 0]], vertices[:, faces[:, 1]], vertices[:, faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = F.normalize(face_normals, dim=-1)

    # Compute vertex normals (area-weighted sum of adjacent face normals)
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):
        vertex_normals.scatter_add_(1, faces[:, i].unsqueeze(-1).expand_as(face_normals), face_normals)
    vertex_normals = F.normalize(vertex_normals, dim=-1)

    # Compute curvature (dot product of vertex normal with neighbor normals)
    curvature = torch.zeros(vertices.shape[:2], device=vertices.device)
    for i in range(3):
        curvature += (vertex_normals[:, faces[:, i]] * face_normals).sum(dim=-1)
    
    curvature = curvature.abs().mean(dim=1)  # Aggregate curvature values
    return curvature



def adaptive_morans_measure(weight, feature, curvature):
    """
    Adaptive Moran's I with curvature adjustment.

    Args:
        weight: B x N x N matrix of weights (computed from query_nn)
        feature: B x N x F matrix of features
        curvature: B x N tensor representing curvature of each Gaussian
    
    Returns:
        moran: Adaptive Moran's I measure
    """
    assert not torch.isnan(weight).any()
    assert not torch.isnan(curvature).any()
    
    B, N, F = feature.shape

    # Normalize curvature (avoid division by zero)
    curvature = curvature.clamp(min=1e-5)  # B x N
    curvature = curvature.unsqueeze(-1)  # B x N x 1 (for broadcasting)

    # Apply curvature-based weight adjustment
    adaptive_weight = weight / curvature  # Higher curvature → Lower weight impact
    adaptive_weight = adaptive_weight / adaptive_weight.sum(-1, keepdim=True).clamp(min=1e-5)  # Normalize

    # Moran's I computation
    W = adaptive_weight.sum(-1).sum(-1)[:, None, None]  # B x 1 x 1
    w_ij = (N / W) * adaptive_weight  # Adaptive weight normalization

    x_mean = feature  # Use raw feature values (or mean-centered if needed)
    denom = (x_mean**2).sum(dim=1)  # B x F

    x_mean_bf = x_mean.contiguous().permute(0, 2, 1).reshape(-1, N)  # B*F x N
    x_corr = (x_mean_bf.unsqueeze(-1) @ x_mean_bf.unsqueeze(-2))  # B*F x N x N
    x_corr = x_corr.view(B, F, N, N)  # B x F x N x N

    nom = (w_ij.unsqueeze(1) * x_corr).sum(-1).sum(-1)  # B x F
    moran = nom / (denom + 1e-4)  # B x F

    return moran.mean()

def adaptive_morans_loss(weight, feature, curvature):
    moran_score = adaptive_morans_measure(weight, feature, curvature)
    moran_loss = 1.0 - moran_score.clamp(0, 1)
    return moran_loss






import torch.nn as nn

# https://github.com/david-svitov/HAHA/blob/main/criteria/sobel_loss.py
class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3: # modified
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)


class SobelLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self._weight = weight
        self._sobel_loss = GradLoss()

    def forward(self, gt_image, render_image):

        device = gt_image.device
        self._sobel_loss.to(device)

        loss = self._sobel_loss(render_image.unsqueeze(0), gt_image.unsqueeze(0)) * self._weight
        return loss