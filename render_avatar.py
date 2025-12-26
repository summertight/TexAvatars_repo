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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os; os.environ["OMP_NUM_THREADS"] = "1"
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
from ae.params_model import SplatDecoder
from ae.view_encoding import plucker, plucker_init, to_cam_matrix, cam_matrix_init
from utils.system_utils import searchForMaxIteration

from utils.loss_utils import ssim, l1_loss
from lpipsPyTorch import lpips
from utils.image_utils import psnr
mesh_renderer = NVDiffRenderer(use_opengl=False)

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh, evaluate):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    if evaluate:
        l1_total, psnr_total, ssim_total, lpips_total = 0.0, 0.0, 0.0, 0.0
        num_frames = len(views_loader)

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        if gaussians.ontop != None:

            cam_matrix = to_cam_matrix(view)
            c2w = (view.world_view_transform.T).inverse().cuda()
            H, W = view.image_height, view.image_width
            
            viewpoint = plucker(cam_matrix['K'].unsqueeze(0), c2w.unsqueeze(0).unsqueeze(0), H, W, "cuda")
            gaussians.select_mesh_by_timestep(view.timestep, viewpoint = viewpoint)

        rendering = render(view, gaussians, pipeline, background, override_color = gaussians.get_colors)['render']

        gt = view.original_image[0:3, :, :]
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
        
        if evaluate:
            l1_total += l1_loss(rendering, gt.cuda()).mean().cpu().item()
            psnr_total += psnr(rendering, gt.cuda()).mean().cpu().item()
            ssim_total += ssim(rendering, gt.cuda()).mean().cpu().item()
            lpips_total += lpips(rendering.unsqueeze(0), gt.cuda().unsqueeze(0)).sum().cpu().item()

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
        
    if evaluate:
        l1_avg = l1_total / num_frames
        psnr_avg = psnr_total / num_frames
        ssim_avg = ssim_total / num_frames
        lpips_avg = lpips_total / num_frames

        print(f"Evaluation Results for {dataset.model_path} \n : L1 {l1_avg:.4f}, PSNR {psnr_avg:.4f}, SSIM {ssim_avg:.4f}, LPIPS {lpips_avg:.4f}")
        
    try:
        import imageio.v2 as imageio
        import os
        from glob import glob

        def save_video_from_images(image_folder, output_path, fps=25):
            image_files = sorted(glob(os.path.join(image_folder, "*.png")))  
            frames = [imageio.imread(img) for img in image_files]  
            imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1) 

        save_video_from_images(render_path, f"{iter_path}/renders_texavatars.mp4")
        save_video_from_images(gts_path, f"{iter_path}/gt.mp4")

        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool, evaluate: bool = False):
    splat_decoder = SplatDecoder(dataset.model_path,
                                 norm_type = dataset.norm_type)
    
    if iteration == -1:
        load_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))

    else:
        load_iter = iteration

    pretrained_path = os.path.join(dataset.model_path,
                                "point_cloud",
                                "iteration_" + str(load_iter),
                                "model_state.pth")

    ckpt = torch.load(pretrained_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if 'plucker_encoder' in k:
            new_k = k.replace('plucker_encoder', 'view_encoder')
        else:
            new_k = k
        new_state_dict[new_k] = v


    splat_decoder.load_state_dict(new_state_dict)
    splat_decoder.eval()

    with torch.no_grad():
        if dataset.on_top: 

            gaussians = FlameGaussianModel(splat_decoder.geom_decoder, splat_decoder.appr_decoder, splat_decoder.exp_encoder, 
                                       dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,
                                       300, 100,
                                       dataset.add_tongue,
                                       uv_res = dataset.uv_res,
                                       sample_res = dataset.sample_res)
                                   
        else:
            gaussians = GaussianModel(dataset.sh_degree)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, org_res=pipeline.org_res, use_mask=pipeline.use_mask)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))

             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, evaluate)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, evaluate)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh, evaluate)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh, evaluate)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh, args.evaluate)