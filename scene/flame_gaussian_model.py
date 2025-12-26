# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
# Modified by TexAvatars

import sys
from utils.uv import gen_tritex, deformation_uv, calculate_tbn_uv
from pathlib import Path
try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj
import numpy as np
import torch
from flame_model.flame import FlameHead
from typing import Literal, List, Union, Dict, Optional, Tuple
import os
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation, compute_face_Jacobian, safe_normalize, compute_vertex_normals
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from torch.nn.functional import grid_sample
import torch.nn.functional as F


class FlameGaussianModel(GaussianModel):
    def __init__(self, geom_decoder, appearance_decoder, exp2code, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100,
                 add_tongue=False, uv_res=512, sample_res=512):
        super().__init__(sh_degree)

        self.geom = geom_decoder.cuda()
        self.appearance = appearance_decoder.cuda()
        self.exp2code = exp2code.cuda()

        print(f"disable_flame_static_offset: {disable_flame_static_offset}")
        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr
        self.uv_res = uv_res

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
            add_tongue=add_tongue,
            uv_res = uv_res,
            sample_res = sample_res
        ).cuda()

        self.flame_param = None
        self.flame_param_orig = None

        self.emo_emb = None

        uv_faces_index, _ = self.sample_uv_map_2(self.flame_model._triim.unsqueeze(0).unsqueeze(0), self.flame_model._uv_grid, n_channels = 1) # justin
        self.uv_faces_index = uv_faces_index.squeeze(0).squeeze(-1).long() # Used in get_* functions in scene/gaussian_model.py

        self.flame_params_optim = None

        if self.ontop is None:
            self.ontop = torch.arange(len(self.flame_model.faces)).cuda()
            self.ontop_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()


        tongue_verts, tongue_faces, tongue_aux = load_obj('./flame_model/assets/flame/tongue_symV2.obj', load_textures=False)
        tongue_faces = tongue_faces.verts_idx

        self.tongue_bary_init_points = tongue_verts[tongue_faces].mean(1)
        self.tongue_attrs = None
        self.N_tongue = 0

        if add_tongue:
            self.tongue_uv_mask = torch.isin(self.uv_faces_index, self.flame_model.mask.f.tongue)
            self.non_tongue_uv_mask = ~torch.isin(self.uv_faces_index, self.flame_model.mask.f.tongue)
            self.tongue_uv_indices = torch.nonzero(self.tongue_uv_mask).squeeze(1)
            self.non_tongue_uv_indices = torch.nonzero(self.non_tongue_uv_mask).squeeze(1)

    def load_extras(self, train_extras, test_extras, tgt_train_extras, tgt_test_extras):
        if self.emo_emb is None:
            extras = {**train_extras, **test_extras}
            tgt_extras = {**tgt_train_extras, **tgt_test_extras}
            pose_extras = extras if len(tgt_extras) == 0 else tgt_extras
            
            self.num_timesteps = max(pose_extras) + 1  
            T = self.num_timesteps
     
            self.emo_emb = torch.zeros([T, 128], dtype=torch.float32, device=torch.device('cuda'))

            for i, extra in pose_extras.items(): 
                self.emo_emb[i] = torch.from_numpy(extra['emo']).float().cuda()
            
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes): 
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1 
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )

        self.update_mesh_position(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False, viewpoint = None, emo_emb = None):
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param
        
        verts, verts_cano, ldmks = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=True,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )

        vertices = verts.squeeze(0).detach().cpu().numpy()  # [N, 3]
        vertices_cano = verts_cano.detach().squeeze(0).cpu().numpy()

        self.ldmks = ldmks

        flame_exp = torch.cat([ flame_param[param][[timestep]].detach() for param in ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose'] ], dim = 1).cuda()
        exp = flame_exp if self.emo_emb is None else torch.cat([flame_exp, self.emo_emb[[timestep]]], -1)

        gs_params = self.geom(exp.detach(), self.exp2code)
        color_params = self.appearance(exp.detach(), self.exp2code, viewpoint) 
        

        self.update_mesh_position_uv(verts, verts_cano)
        self.update_gs_offsets_uv(verts[0], gs_params, color_params)

    

    def convert_3D_to_uv(self, vert_values, return_type = 'uv'):
        """
        Closely related to flame_model/flame_model.py
        Args:
            vert_values: It could be any vertex anchored values (vertex position, etc.,) and size :(N, F) 
         
        Returns: (H, W, F) : Texel map
        
        """
        assert return_type in ['uv','uvd']

        v0_map = vert_values[self.flame_model._idxim[..., 0]] 
        v1_map = vert_values[self.flame_model._idxim[..., 1]] 
        v2_map = vert_values[self.flame_model._idxim[..., 2]] 


        flame_uv_map = self.flame_model._barim[..., [0]] * v0_map + \
                                 self.flame_model._barim[..., [1]] * v1_map + \
                                     self.flame_model._barim[..., [2]] * v2_map
                                     # Apply barycentric interpolation
        torch_uv_map = flame_uv_map.float().permute(2, 0, 1)


        return torch_uv_map


    def update_gs_offsets_uv(self, deformed_verts, gs_params, color_params): 
        """
        Closely related to flame_model/flame_model.py
        Args:
            deformed_verts: Deformed vertices after forwarding of FLAME model.
            gs_params: Regressed 2D gaussian (geometry + opacity) map.
            color_params: Regressed 2D gaussian (rgb) map.
         
        Returns: No return - Update gaussian (offset) parameters with the regressed gaussian maps using UV mapping.
        
        """

        self.torch_position_map = self.convert_3D_to_uv(deformed_verts)
        self.torch_normal_map = self.convert_3D_to_uv(self.vertex_normals[0]) 
        B, C, H, W = gs_params.shape
        N = H * W

        self._xyz, self._rotation, self._scaling, self._opacity = torch.split(gs_params, [3, 4, 3, 1], dim=1)
        self._color = color_params
        
        triim = self.flame_model._triim.long()  # (512, 512)
        valid_mask = self.flame_model.valid_mask
        safe_triim = triim.clone() #! 512**2
        safe_triim[~valid_mask] = 0

        Jacobian_mat_uv = self.face_Jacobian_mat[safe_triim] #! 512**2 3x3
        Jacobian_mat_uv[~valid_mask] = torch.eye(3).to(Jacobian_mat_uv)
    
        _xyz_uv = self._xyz.permute(2,3,1,0).contiguous()
        position_uv = torch.einsum(" HWij, HWjk -> HWik", Jacobian_mat_uv, _xyz_uv).squeeze(-1) + self.torch_position_map.permute(1,2,0).cuda()
        position_uv = position_uv[None].permute(0,3,1,2).contiguous() #! 3:X
        opacity_uv = self._opacity
        color_uv = self._color

        scaling_log_uv = self._scaling[0].permute(1,2,0).contiguous() #! 3:s
        rot_unnorm_uv = self._rotation[0].permute(1,2,0).contiguous() #! 4:Q
        covariance_uv = self.covariance_activation(self.scaling_activation(scaling_log_uv).reshape(N, 3), 1, rot_unnorm_uv.reshape(N, 4), Jacobian_mat_uv.reshape(N, 3, 3)).reshape(H, W, 6) 

        payloads_uv = torch.cat([covariance_uv[None].permute(0,3,1,2), position_uv, color_uv, opacity_uv], dim = 1).contiguous()
        payloads = self.sample_uv_map(payloads_uv, self.flame_model._uv_grid.detach(), n_channels = 13)[0].squeeze(0)

        self.covariance_final, position_final, color_final, opacity_final = \
                        torch.split(payloads, [6,3,3,1], dim=1)
        self.color_final = color_final
        self.opacity_final = self.opacity_activation(opacity_final)
        self.position_final = position_final


    def sample_uv_map(self, planes: torch.Tensor, uv_grid: torch.Tensor, n_shells: int = 1, n_channels: int = 11): 
        """
        For sampling only valid gaussians (located in faces in uv map) from regressed 2D gaussian maps.
        """
        B, C, H_f, W_f = planes.shape
        S = n_shells
        C_uv = n_channels

        uv_map = planes
        uv_map = uv_map.reshape(B * S, C_uv, H_f, W_f) 

        uv_attributes = grid_sample(uv_map, uv_grid,
                                    align_corners=False,
                                    mode='bilinear') 
        uv_attributes = uv_attributes.squeeze(3).permute(0, 2, 1) 
        G = uv_attributes.shape[1]

        uv_attributes = uv_attributes.reshape(B, S * G, C_uv)

        return uv_attributes, uv_map

    def sample_uv_map_2(self, planes: torch.Tensor, uv_grid: torch.Tensor, n_shells: int = 1, n_channels: int = 11):
        """
        Please search for [UV Map for Triangle Properties] keyword.
        """
        B, C, H_f, W_f = planes.shape
        S = n_shells
        C_uv = n_channels

        uv_map = planes

        uv_map = uv_map.reshape(B * S, C_uv, H_f, W_f)
        uv_attributes = grid_sample(uv_map, uv_grid,
                                    align_corners=False,
                                    mode='nearest') 
        
        uv_attributes = uv_attributes.squeeze(3).permute(0, 2, 1)  
        G = uv_attributes.shape[1]

        uv_attributes = uv_attributes.reshape(B, S * G, C_uv)
        return uv_attributes, uv_map
    

    def _collect_gaussian_attributes(self, predictions: torch.Tensor,
                                     att_names = ['dc', 'rest'], channels = [3, 45], return_raw_attributes: bool = False):
        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()
        c = 0
        # breakpoint()
        for ch in zip(att_names, channels):
            n_channels = ch[1]

            attribute_map = predictions[..., c: c + n_channels]  # Slice corresponding channels from sampled plane
            if return_raw_attributes:
                raw_gaussian_attributes[ch[0]] = attribute_map

            gaussian_attributes[ch[0]] = attribute_map
            c += n_channels

        return gaussian_attributes, raw_gaussian_attributes


    def update_mesh_position_uv(self, verts, verts_cano): 
        
        faces = self.flame_model.faces 
        triangles = verts[:, faces] # Now 'traingles' variable consists of deformed 3D vertex coordinates.

        self.face_center = triangles.mean(dim = -2).squeeze(0) # legacy
        self.face_Jacobian_mat = compute_face_Jacobian(verts.squeeze(0), faces.squeeze(0))
        
        self.vertex_normals = compute_vertex_normals(verts, faces.squeeze(0)) #! 5143, 3
        self.verts = verts
        self.faces = faces

        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):

        if self.not_finetune_flame_params:
            return

        l = []

        for k in ['rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation', 'expr']:
            self.flame_param[k] = self.flame_param[k].detach().requires_grad_()

        # pose
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        l.append(param_pose)

        # translation
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        l.append(param_trans)

        # expression
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        l.append(param_expr)

        self.flame_params_optim = l


    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:

            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:

            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
