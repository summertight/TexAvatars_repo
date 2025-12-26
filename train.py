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
from torch.optim.lr_scheduler import StepLR
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os; os.environ["OMP_NUM_THREADS"] = "1"
import copy
import torch

import matplotlib.cm as cm
import torch.nn.functional as F
from random import randint
from torch.utils.data import DataLoader
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, error_map
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from ae.params_model import SplatDecoder 
from ae.view_encoding import plucker, plucker_init, to_cam_matrix, cam_matrix_init
from utils.extract_geo import morans_measure, query_nn, morans_loss, neg_morans_loss, adaptive_morans_loss, compute_curvature, SobelLoss
from mesh_renderer import NVDiffRenderer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    mesh_renderer = NVDiffRenderer()
except:
    mesh_renderer = None

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    splat_decoder = SplatDecoder(dataset.model_path,
                                 norm_type = dataset.norm_type)

    
    gaussians = FlameGaussianModel(splat_decoder.geom_decoder, splat_decoder.appr_decoder, splat_decoder.exp_encoder, 
                                    dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,
                                    300, 100,
                                    dataset.add_tongue,
                                    uv_res = dataset.uv_res,
                                    sample_res = dataset.sample_res)
    
    scene = Scene(dataset, gaussians, org_res=pipe.org_res, use_mask=pipe.use_mask)

    if opt.lambda_vgg > 0:
        from vggPyTorch.vgg19 import VGGLoss
        vgg = VGGLoss().cuda()

    splat_decoder.training_setup(opt)
    gaussians.training_setup(opt) # Turn on FLAME-finetuning
    if not dataset.not_finetune_flame_params:
        for flame_param in gaussians.flame_params_optim: # Incorporating FLAME parameters to learnable parameres becomes tricky here
            splat_decoder.optimizer.add_param_group(flame_param) # Optimizer in splat decoder must be the only optimizer
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        splat_decoder.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)



    for iteration in range(first_iter, opt.iterations + 1): 

        iter_start.record()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        gt_image = viewpoint_cam.original_image.cuda()


        if gaussians.ontop != None:

            import numpy as np
            cam_matrix = to_cam_matrix(viewpoint_cam)
            c2w = (viewpoint_cam.world_view_transform.T).inverse().cuda()
            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
            
            viewpoint = plucker(cam_matrix['K'].unsqueeze(0), c2w.unsqueeze(0).unsqueeze(0), H, W, "cuda")

            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep, viewpoint = viewpoint, emo_emb = gaussians.emo_emb)

        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, viewpoint_cam)
        rgb_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)[:3, :, :]
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color = gaussians.get_colors)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        losses = {}
        
            
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

        if opt.vgg_kickin < iteration:
            losses['vgg'] = vgg(image[None], gt_image[None]) * opt.lambda_vgg

        if gaussians.ontop != None:
            N = gaussians._xyz.shape[0]
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:

                losses['xyz'] = F.relu(gaussians._xyz.norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz


            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian


        losses['total'] = sum([v for k, v in losses.items()])
        if dataset.not_finetune_flame_params:
            losses['total'].backward()
        else:
            losses['total'].backward(retain_graph=True) 

        iter_end.record()

        with torch.no_grad():

            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:

                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                for _k, _v in losses.items():
                    if _k == 'total':
                        continue
                    postfix[_k] = f"{_v:.{7}f}"

                progress_bar.set_postfix(postfix)

                losses_for_log = copy.deepcopy(postfix)
                losses_for_log['l1'] = f"{losses['l1']:.{7}f}"
                losses_for_log['ssim'] = f"{losses['ssim']:.{7}f}"

                with open(dataset.model_path + "/log.txt", "a") as file:
                    file.write(f"{iteration}\t")
                    file.write("\t".join([ f"{key}: {value} " for key, value in losses_for_log.items()]))
                    file.write("\n")


                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1.0), dataset.model_path)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                
                scene.save(iteration)
                splat_decoder.save(iteration) 

            # DENSIFICATION REMOVED

            if iteration < opt.iterations:

                splat_decoder.optimizer.step()
                splat_decoder.optimizer.zero_grad(set_to_none = True)
                
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log

                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, log_path): # justin
    if tb_writer:
        for _k, _v in losses.items():
            tb_writer.add_scalar(f'train_loss_patches/{_k}', _v.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0


                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if scene.gaussians.num_timesteps > 1:
                        
                        # [View Direction] 

                        # viewpoint_val = torch.tensor(viewpoint.R, dtype=torch.float32, device = "cuda").T # justin
                        cam_matrix = to_cam_matrix(viewpoint)
                        c2w = (viewpoint.world_view_transform.T).inverse().cuda()
                        H, W = viewpoint.image_height, viewpoint.image_width

                        viewpoint_val = plucker(cam_matrix['K'].unsqueeze(0), c2w.unsqueeze(0).unsqueeze(0), H, W, "cuda")
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep, viewpoint = viewpoint_val)

                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, override_color=scene.gaussians.get_colors)
                    image = torch.clamp(render_pkg['render'], 0.0, 1.0)

                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                            
                            if mesh_renderer is not None:
                                out_dict = mesh_renderer.render_from_camera(scene.gaussians.verts, scene.gaussians.faces, viewpoint)
                                rgb_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)[:3, :, :]
                                tb_writer.add_images(config['name'] + "_{}/FLAME_mesh".format(vis_ct), rgb_mesh[None], global_step=iteration)

                        vis_ct += 1

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))

                with open(log_path + "/log.txt", "a") as file:
                    file.write("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                    file.write("\n")


                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--emoemb_id', type=str, required = False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default = [])
    parser.add_argument("--save_iterations", nargs="+", type=int, default = [])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")