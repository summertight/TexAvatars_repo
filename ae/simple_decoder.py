import math
import torch
from torch import nn
import torch.nn.functional as F

from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import ae.nn.layers as la
from ae.nn.layers import make_conv_trans, make_linear, make_conv

# Modified from RGCA (https://github.com/facebookresearch/goliath)
    
class Exp2Code(nn.Module):
    def __init__(self, use_emo=False):
        super().__init__()
        exp_dim = 100 + 5 * 3 + 128

        self.encmod = nn.Sequential(
            *make_linear(exp_dim, 256 * 8 * 8, "wn", nn.LeakyReLU(0.2, inplace=True))
        )

    def forward(self, flame_exp):
        return self.encmod(flame_exp)
    
class Plucker2Code(nn.Module):
     '''
     Plucker2Code
     '''
     def __init__(self, view_dim = 8, norm_type="wn"):
        super().__init__()
        assert norm_type in ["wn", "sn"]

        self.viewmod = nn.Sequential(
            *make_conv(6, 16, 3, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True)), # to 64x64
            *make_conv(16, 32, 3, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True)), # to 32x32
        )
 
     def forward(self, plucker):
         plucker_square =F.interpolate(plucker.squeeze(2), size=(512, 512), mode='bilinear', align_corners=False)
 
         plucker_emb = F.interpolate(plucker_square, size=(256, 256), mode='bilinear', align_corners=False)
         plucker_emb = F.interpolate(plucker_emb, size=(128, 128), mode='bilinear', align_corners=False)
         plucker_emb = self.viewmod(plucker_emb)   
         return plucker_emb  

class R2Code(nn.Module):

    def __init__(self, view_dim = 8): 
        super().__init__()
        self.viewmod = nn.Sequential(
            *make_linear(9, view_dim, "wn", nn.LeakyReLU(0.2, inplace=True)) 
        )

    def forward(self, cam_R):
        return self.viewmod(cam_R)  

class GeomDecoder(nn.Module):
    def __init__(self, output_dim, norm_type="wn", from_const=False):
        super().__init__()
        assert norm_type in ["wn", "sn"]
        self.output_dim = output_dim 
        self.code2geom = nn.Sequential(
                *make_conv_trans(
                    256, 256, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(16, 16)
                ),
                *make_conv_trans(
                    256, 128, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(32, 32)
                ),
                *make_conv_trans(
                    128, 128, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(64, 64)
                ),
                *make_conv_trans(
                    128, 64, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(128, 128)
                ),
                *make_conv_trans(
                    64, 32, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)
                ),
                *make_conv_trans(
                    32, self.output_dim, 4, 2, 1, norm_type, ub=(512, 512) 
                ),
            )
        
    def forward(
        self,
        flame_exp: torch.Tensor,
        exp2code: Exp2Code,
    ):

        exp_code = exp2code(flame_exp).view(-1, 256, 8, 8)

        idx = 0
        geom = exp_code

        for layer in self.code2geom:

            geom = layer(geom)
            idx += 1

        return geom


class ApprDecoder(nn.Module):

    def __init__(self, output_dim=48, norm_type="wn", use_plucker = False):
        super().__init__()
        assert norm_type in ["wn", "sn"]
        self.output_dim = output_dim
        self.view_out_dim = 32
        self.view_encoder = Plucker2Code(self.view_out_dim, norm_type=norm_type)

        self.appr_decoder = nn.ModuleList([
            *make_conv_trans(
                256,
                256,
                4,
                2,
                1,
                norm_type,
                nn.LeakyReLU(0.2, inplace=True),
                ub=(16, 16),
            ),
            *make_conv_trans(
                256, 128, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(32, 32)
            ),

            *make_conv      (
                128 + 32, 128, 3, 1, 1, norm_type, nn.LeakyReLU(0.2, inplace=True),
            ),
            *make_conv      (
                128, 128, 3, 1, 1, norm_type, nn.LeakyReLU(0.2, inplace=True),
            ),

            *make_conv_trans(
                128, 64, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(64, 64) 
            ),
            *make_conv_trans(
                64, 64, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(128, 128)
            ),

            *make_conv_trans(
                64, self.output_dim, 4, 2, 1, norm_type, nn.LeakyReLU(0.2, inplace=True), ub=(256, 256)
            ),
            *make_conv_trans(
                self.output_dim, self.output_dim, 4, 2, 1, norm_type, ub = (512, 512)
            )

        ])
                    
    def forward(
        self,
        flame_exp: torch.Tensor,
        exp2code: Exp2Code,
        view_direction: torch.Tensor
        ):

        exp_code = exp2code(flame_exp).view(-1, 256, 8, 8) 
        view_direction = view_direction.cuda()

        view_emb = self.view_encoder(view_direction)

        appr = exp_code
        idx = 0

        for layer in self.appr_decoder:

            if idx == 4:
                appr = torch.cat([appr, view_emb], dim = 1)     

            appr = layer(appr)
            idx += 1

        return appr
    



    
    