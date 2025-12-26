import torch
from torch import nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import os
from ae.simple_decoder import GeomDecoder, ApprDecoder, Exp2Code

class SplatDecoder(nn.Module):
    def __init__(self, model_path, norm_type="wn"):
        
        super(SplatDecoder, self).__init__()
        self.model_path = model_path

        self.exp_encoder = Exp2Code()

        
        geom_dim = 11
        self.geom_decoder = GeomDecoder(output_dim=geom_dim, norm_type=norm_type)

        color_dim = 3
        self.appr_decoder = ApprDecoder(output_dim=color_dim, norm_type=norm_type) # internally containing Plucker2Code

        self.optimizer = None
        self.percent_dense = None
        self.xyz_gradient_accum = None
        self.denom = None

    def capture(self):
        return self.optimizer.state_dict()
    
    def training_setup(self, training_args):

        l = [
            {'params': self.geom_decoder.parameters(), 'lr': training_args.geom_lr, "name": "geom"},
            {'params': self.appr_decoder.parameters(), 'lr': training_args.appr_lr, "name": "appr"}, 
            {'params': self.exp_encoder.parameters(), 'lr': training_args.exp2code_lr, "name": "exp2code"},
        ]

        self.optimizer = torch.optim.Adam(l, lr = 0.0, eps = 1e-15)

    def save(self, iteration):
        iter_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.save_params(iter_path)

    def save_params(self, save_path):

        model_save_path = os.path.join(save_path, "model_state.pth")
        torch.save(self.state_dict(), model_save_path)

        if self.optimizer is not None:
            optimizer_save_path = os.path.join(save_path, "optimizer_state.pth")
            torch.save(self.optimizer.state_dict(), optimizer_save_path)