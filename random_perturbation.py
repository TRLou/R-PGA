from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import torch
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
import sys
import copy

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
import torchvision
from envlight.utils import cubemap_to_latlong
from train_func import (
    DETECTOR_PATHS,
    render_and_save_final_images,
    evaluate_from_saved_images,
    latest_checkpoint_pth,
)
from utils.main_utils import coco_classes
from mmdet.apis import init_detector

def get_perturb_args():
    """Defines and parses command-line arguments for the random perturbation evaluation."""
    parser = argparse.ArgumentParser("RGA Random Perturbation Evaluation")
    
    # Add model and pipeline parameters from the original arguments file
    model_params = ModelParams(parser, sentinel=False)
    pipeline_params = PipelineParams(parser)

    # --- Script-Specific Configuration ---
    parser.add_argument('--num_runs', type=int, default=5, help="Number of random perturbation runs to average.")

    # --- Model & Scene Loading ---
    parser.add_argument('--iteration', type=int, default=-1, help='Iteration to load; -1 means latest (GIR style).')
    parser.add_argument('--second_stage_step', type=int, default=30000, help="Step count for the second stage of rendering.")
    
    # --- Detector Configuration ---
    parser.add_argument('--detector', type=str, default='yolox', 
                        choices=list(DETECTOR_PATHS.keys()),
                        help="MMDetection model to use if --all_detectors is not set.")
    parser.add_argument('--all_detectors', default=True, action=argparse.BooleanOptionalAction,
                        help="Evaluate on all available detectors (default: True). If set, --detector is ignored.")
    parser.add_argument('--target_class_name', type=str, default='car', help="Target class name for the attack (COCO class).")
    parser.add_argument('--score_thresh', type=float, default=0.5, help="Score threshold for detection.")

    # --- Environment ---
    parser.add_argument('--environment_texture', type=str, default="", help="Path to the environment texture (HDR).")
    parser.add_argument('--environment_scale', type=float, default=1.0, help="Scale of the environment light.")
    
    # --- Albedo Perturbation ---
    parser.add_argument('--perturb_budget_factor', type=float, default=1, help='Factor (n) for calculating the albedo perturbation budget.')
    parser.add_argument('--albedo_init_method', type=str, default='perturb', choices=['perturb', 'random'], help='Albedo initialization method.')

    # --- I/O and System ---
    parser.add_argument('--save_dir', type=str, default='RGA_perturb_output', help="Directory to save outputs.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the training on.")

    parser.add_argument('--hdr_rotation', action='store_true', default=False, help="Enable random rotation of the HDR environment map during training.")

    args = get_combined_args(parser)
    
    # Hardcode some values that are not relevant for this script
    args.epochs = 1 
    args.lr = 0.0 
    
    return args, model_params, pipeline_params

def main():
    # =================================================================================
    # 1. 参数解析与环境设置
    # =================================================================================
    args, model_params, pipeline_params = get_perturb_args()

    if not args.source_path or not args.model_path:
        print("\n[错误] 必须通过 --source_path 和 --model_path 提供数据和模型路径。")
        print("用法示例: python -m RGA.random_perturbation -s /path/to/dataset -m /path/to/model\n")
        sys.exit(1)

    device = torch.device(args.device)
    
    # Create a timestamped directory for all outputs
    save_dir_base = Path(args.save_dir)
    if not save_dir_base.is_absolute():
        repo_dir = Path(__file__).resolve().parent
        save_dir_base = repo_dir / save_dir_base
        
    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%m%d_%H%M%S") + "_Beijing_Perturb"
    save_dir = save_dir_base / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[消息] 所有输出将保存到: {save_dir}")

    # Save command line arguments
    with open(save_dir / "args.txt", 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    # =================================================================================
    # 2. 加载场景与原始高斯模型
    # =================================================================================
    dataset = model_params.extract(args)
    
    # Correct Initialization Order (same as train_rel_attack.py)
    # 1. Create a GaussianModel instance.
    if args.environment_texture:
        gaussians = GaussianModel(dataset.sh_degree, environment_texture=args.environment_texture, environment_scale=args.environment_scale)
    else:
        gaussians = GaussianModel(dataset.sh_degree)

    # 2. Load envlight from checkpoint BEFORE initializing the Scene.
    #    This ensures that when the Scene loads the point cloud data, the envlight is already present.
    if not args.environment_texture:
        print("[消息] 未提供 environment_texture，将尝试从最新的检查点加载 'envlight'...")
        model_dir = Path(dataset.model_path)
        latest_ckpt_path = latest_checkpoint_pth(model_dir)
        if latest_ckpt_path:
            print(f"[消息] 找到最新检查点: {latest_ckpt_path}")
            try:
                ckpt_data_tuple, _ = torch.load(str(latest_ckpt_path), map_location=device)
                envlight_state_dict = None
                for item in ckpt_data_tuple:
                    if isinstance(item, dict) and not ('state' in item and 'param_groups' in item):
                        envlight_state_dict = item
                        break
                
                if envlight_state_dict:
                    print("[消息] 正在从检查点加载 'env_light'...")
                    gaussians.envlight.load_state_dict(envlight_state_dict)
                    print("[消息] 成功加载 'envlight'。")
                    try:
                        print("[消息] 正在保存加载的 envlight 到图片以供验证...")
                        hdr_image = cubemap_to_latlong(gaussians.envlight.base.detach(), [512, 1024]).permute(2,0,1).contiguous()
                        verify_path = save_dir / 'loaded_envlight_from_ckpt.png'
                        torchvision.utils.save_image(hdr_image.cpu(), str(verify_path))
                        print(f"[消息] 验证图片已保存到: {verify_path}")
                    except Exception as e:
                        print(f"[消息] 保存验证图片失败: {e}")
                else:
                    print("[警告] 在检查点中未找到 'env_light' 的 state_dict。")
            except Exception as e:
                print(f"[错误] 从检查点加载 envlight 失败: {e}")
        else:
            print("[警告] 未找到 '.pth' 检查点文件。场景可能渲染为黑色。")

    # 3. Now, initialize the Scene. It will load point cloud data into the `gaussians` object
    #    that already contains the loaded envlight.
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # 4. Finally, compute occlusion-related buffers.
    try:
        gaussians.get_diffuse_occ()
        print("[消息] 已计算并初始化高斯模型的遮挡相关缓冲区。")
    except Exception as e:
        print(f"[警告] 计算遮挡相关缓冲区失败: {e}")
    
    # Keep a clean, original copy of the gaussians
    gaussians_original = copy.deepcopy(gaussians)
    print("[消息] 原始高斯模型已加载并备份。")

    # =================================================================================
    # 3. 准备评估
    # =================================================================================
    all_cameras = scene.getTrainCameras()
    random.shuffle(all_cameras)
    split_idx = int(len(all_cameras) * 0.9)
    train_cameras = all_cameras[:split_idx]
    test_cameras = all_cameras[split_idx:]
    full_cameras = train_cameras + test_cameras
    
    pipe = pipeline_params.extract(args)
    bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)

    if args.all_detectors:
        detector_names = list(DETECTOR_PATHS.keys())
    else:
        detector_names = [args.detector]
    print(f"[消息] 将在以下检测器上进行评估: {', '.join(detector_names)}")
    
    # Data structure to store results from all runs
    all_run_results = defaultdict(lambda: defaultdict(list))

    # =================================================================================
    # 4. 开始多次随机扰动评估循环
    # =================================================================================
    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'='*40}\n[消息] 开始第 {run_idx}/{args.num_runs} 次随机扰动评估\n{'='*40}")
        run_save_dir = save_dir / f"run_{run_idx:02d}"
        run_save_dir.mkdir(parents=True, exist_ok=True)

        # a. Create a fresh working copy of the gaussians model
        gaussians = copy.deepcopy(gaussians_original)

        # b. Perturb albedo by first zeroing it, then adding a random value in [-0.5, 0.5]
        print("[消息] 正在对 Albedo 进行置零并添加 [-0.5, 0.5] 的随机扰动...")
        with torch.no_grad():
            # Zero out the albedo first
            gaussians._albedo_init.data.zero_()
            
            # Generate and add random perturbation in [-0.5, 0.5]
            perturbation = torch.rand_like(gaussians._albedo_init.data) - 0.5  # This creates values in [-0.5, 0.5]
            gaussians._albedo_init.data += perturbation
            
            # Clamp to ensure it stays strictly within [-0.5, 0.5]
            gaussians._albedo_init.data.clamp_(-0.5, 0.5)
            
            perturbed_albedo = gaussians._albedo_init.data
            print(f"[消息] Albedo 已扰动. 新范围: [{perturbed_albedo.min().item():.4f}, {perturbed_albedo.max().item():.4f}]")
        
        # c. Render final images for this run
        print("[消息] 正在渲染扰动后的图像...")
        final_test_img_dir = render_and_save_final_images(
            test_cameras, gaussians, pipe, bg, args, dataset, 
            gaussians_original, None, run_save_dir, 'test'
        )
        final_full_img_dir = render_and_save_final_images(
            full_cameras, gaussians, pipe, bg, args, dataset, 
            gaussians_original, None, run_save_dir, 'full'
        )

        # d. Evaluate on all specified detectors
        print("[消息] 正在对渲染图像进行多检测器评估...")
        base_path = Path('/workspace/RGA/mmdet_files')
        
        for det_name in detector_names:
            print(f"  -> 评估检测器: {det_name}")
            try:
                cfg_path = str(base_path / DETECTOR_PATHS[det_name]['config'])
                ckpt_path = str(base_path / DETECTOR_PATHS[det_name]['ckpt'])
                curr_detector = init_detector(cfg_path, ckpt_path, device=device)
                if not hasattr(curr_detector, 'CLASSES'):
                    curr_detector.CLASSES = coco_classes
                
                # Test set evaluation
                asr_test, _, _, ap50_test = evaluate_from_saved_images(
                    curr_detector, final_test_img_dir, Path(dataset.source_path) / 'annos', args
                )
                all_run_results[det_name]['asr_test'].append(asr_test)
                all_run_results[det_name]['ap50_test'].append(ap50_test)

                # Full set evaluation
                asr_full, _, _, ap50_full = evaluate_from_saved_images(
                    curr_detector, final_full_img_dir, Path(dataset.source_path) / 'annos', args
                )
                all_run_results[det_name]['asr_full'].append(asr_full)
                all_run_results[det_name]['ap50_full'].append(ap50_full)

                del curr_detector
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                print(f"[错误] 评估检测器 {det_name} 时发生错误: {e}")

    # =================================================================================
    # 5. 计算并报告平均结果
    # =================================================================================
    print(f"\n{'='*40}\n[消息] {args.num_runs} 次运行的平均评估结果\n{'='*40}")
    
    summary_log_path = save_dir / "summary_results.txt"
    with open(summary_log_path, 'w', encoding='utf-8') as f:
        f.write(f"Random Perturbation Evaluation Summary ({args.num_runs} runs)\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        for det_name, results in sorted(all_run_results.items()):
            avg_asr_test = np.mean(results['asr_test'])
            std_asr_test = np.std(results['asr_test'])
            avg_ap50_test = np.mean(results['ap50_test'])
            std_ap50_test = np.std(results['ap50_test'])

            avg_asr_full = np.mean(results['asr_full'])
            std_asr_full = np.std(results['asr_full'])
            avg_ap50_full = np.mean(results['ap50_full'])
            std_ap50_full = np.std(results['ap50_full'])
            
            header = f"Detector: {det_name}"
            print(header)
            f.write(f"{header}\n")
            
            line1 = f"  - [Test Set] Avg ASR: {avg_asr_test:.4f} (±{std_asr_test:.4f}), Avg AP@0.5: {avg_ap50_test:.4f} (±{std_ap50_test:.4f})"
            print(line1)
            f.write(f"{line1}\n")

            line2 = f"  - [Full Set] Avg ASR: {avg_asr_full:.4f} (±{std_asr_full:.4f}), Avg AP@0.5: {avg_ap50_full:.4f} (±{std_ap50_full:.4f})"
            print(line2)
            f.write(f"{line2}\n")
            
            f.write("-" * 30 + "\n")

    print(f"\n[消息] 平均结果已保存到: {summary_log_path}")

if __name__ == '__main__':
    main()
