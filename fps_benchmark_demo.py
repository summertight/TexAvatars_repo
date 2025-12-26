import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.viewer_utils import OrbitCamera
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel, FlameGaussianModel
import os
from ae.params_model import SplatDecoder#, SplatTeethDecoder
from ae.view_encoding import view_direction, view_direction_2, view_direction_init, plucker, plucker_init, to_cam_matrix, cam_matrix_init
from utils.system_utils import searchForMaxIteration

@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False

def prepare_camera(width, height):
    cam = OrbitCamera(width, height, r=1, fovy=20, convention='opencv')

    @dataclass
    class Cam:
        FoVx = float(np.radians(cam.fovx))
        FoVy = float(np.radians(cam.fovy))
        image_height = cam.image_height
        image_width = cam.image_width
        world_view_transform = torch.tensor(cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
        full_proj_transform = torch.tensor(cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
        camera_center = torch.tensor(cam.pose[:3, 3]).cuda()
    return Cam

def render_sets(pipeline : PipelineConfig, model_path, point_path, sh_degree, height, width, n_iter, vis=False):
    splat_decoder = SplatDecoder(model_path)    
    
    if True:
        load_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

    else:
        load_iter = iteration

    print(f"load_iter: {load_iter}")

    pretrained_path = os.path.join(model_path,
                                "point_cloud",
                                "iteration_" + str(load_iter),
                                "model_state.pth")

    splat_decoder.load_state_dict(torch.load(pretrained_path))
    splat_decoder.eval()
    
    with torch.no_grad():
        # init gaussians
        if (Path(pretrained_path).parent / "flame_param.npz").exists():
            gaussians = FlameGaussianModel(splat_decoder.geom_decoder, splat_decoder.appr_decoder, splat_decoder.exp_encoder, sh_degree)
        else:
            gaussians = GaussianModel(sh_degree)

        # load gaussians
        assert point_path is not None
        if point_path.exists():
            print(point_path)
            gaussians.load_ply(str(point_path) + f"/iteration_{load_iter}/point_cloud.ply", has_target=False)
        else:
            raise FileNotFoundError(f'{point_path} does not exist.')

        background = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
        cam = prepare_camera(width, height)

        for i in range(3):
            print(f"\nRound {i+1}")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in tqdm(range(n_iter)):
                if gaussians.ontop != None:

                    viewpoint_0th = plucker_init() 
                    gaussians.select_mesh_by_timestep(0, viewpoint = viewpoint_0th)
                rendering = render(cam, gaussians, pipeline, background)["render"]
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000
            print(f"Rendering {n_iter} images took {elapsed_time:.2f} s")
            print(f"FPS: {n_iter / elapsed_time:.2f}")

        if vis:
            print("\nVisualizing the rendering result")
            plt.imshow(rendering.permute(1, 2, 0).clip(0, 1).cpu().numpy())
            plt.show()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--point_path", default="output/074_uv_unified_indep_plucker2_STABLE/point_cloud", type=Path)
    parser.add_argument("--model_path", default="output/074_uv_unified_indep_plucker2_STABLE", type=Path)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--height", default=802, type=int)
    parser.add_argument("--width", default=550, type=int)
    parser.add_argument("--n_iter", default=500, type=int)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    print("Rendering " + str(args.point_path))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(PipelineConfig(), args.model_path, args.point_path, args.sh_degree, args.height, args.width, args.n_iter, args.vis)