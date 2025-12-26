#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from ae.view_encoding import plucker_init
from utils.helpers import interp_face_scale, interp_face_mat, interp_face_quat, interp_face_xyz  
from roma import unitquat_to_rotvec, rotvec_to_unitquat

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, Jacobian=None):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if Jacobian is not None:
                actual_covariance = Jacobian @ actual_covariance @ Jacobian.transpose(1, 2)
            
            symm = strip_symmetric(actual_covariance)
            return symm
        
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.shadow_activation = lambda x: 2.0 * torch.sigmoid(x)


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        ## justin ##
        self.ontop = None # whether modeling 2dgs parameter offsets or not
        self.timestep = None  # the current timestep
        self.num_timesteps = 1  # required by viewers
        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        self.uv_faces_index = None # !!!
        self.XYZ = None

        self.use_plucker = None

        
        self.face_scaling_resized = None
        self.use_shadow = False
        self.use_Jacobian = False
        ## jesse ##
        # self.

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    
    @property
    def get_scaling(self):
        
        ## justin ##
        if self.ontop is None:
            return self.scaling_activation(self._scaling)
        else:
            if self.face_scaling is None:

                viewpoint_0th = plucker_init()
                self.select_mesh_by_timestep(0, viewpoint = viewpoint_0th)

            return self.scaling_final            

    
    @property
    def get_rotation(self):

        ## justin ##
        if self.ontop is None:
            return self.scaling_activation(self._rotation)
        else:
            if self.rotation_activation is None:

                viewpoint_0th = plucker_init() 
                self.select_mesh_by_timestep(0, viewpoint = viewpoint_0th) 
            
            if self.uv_operations:
                return self.rotation_final
              

    @property
    def get_xyz(self):
        if self.ontop is None:
            return self._xyz
        else:
            if self.face_center is None:

                viewpoint_0th = plucker_init()
                self.select_mesh_by_timestep(0, viewpoint = viewpoint_0th) 
     
            return self.position_final

 
    @property
    def get_colors(self):
        
        return self.color_final

    
    @property
    def get_features(self):
        features_dc = self.sample_uv_map(self._features_dc, self.flame_model._uv_grid.detach(), n_channels = 3)
        features_rest = self.sample_uv_map(self._features_rest, self.flame_model._uv_grid.detach(), n_channels = 45)

        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_final
    
    @property
    def set_rotation(self, rotation):
        self._rotation = rotation

    @property
    def set_scaling(self, scaling):
        self._scaling = scaling        
    
    @property
    def set_xyz(self, xyz):
        self._xyz = xyz
    
    @property
    def set_features(self, dc, rest):
        self._features_dc = dc
        self._features_rest = rest
    
    @property
    def set_opacity(self):
        return self.opacity_activation(self._opacity)    
    
    def get_covariance(self, scaling_modifier = 1):
        
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)
        
    def get_covariance_Jacobian(self, scaling_modifier = 1, Jacobian=None):
        return self.covariance_final

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : Optional[BasicPointCloud], spatial_lr_scale : float):

        self.spatial_lr_scale = spatial_lr_scale
        if pcd == None:
            assert self.ontop is not None
            num_pts = self.ontop.shape[0] # number of faces

            fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda()
            fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        if self.ontop is None:
            dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) #3dgs
        else:
            scales = torch.log(torch.ones((self.get_xyz.shape[0], 3), device="cuda")) #3dgs
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

    # def reset_opacity(self):
    #     opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors