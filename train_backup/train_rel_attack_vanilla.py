from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import torch
from tqdm import tqdm
import random

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
import torchvision
from envlight.utils import cubemap_to_latlong
from utils.main_utils import coco_classes
from mmdet.apis import init_detector
from lbm_relit import LBMRelighter
from utils.log_utils import TrainingLogger
from attack_options import get_attack_args

from train_func import (
    DETECTOR_PATHS,
    save_visualization_grid,
    compute_batch_loss,
    evaluate,
    render_and_save_final_images,
    evaluate_from_saved_images,
    load_hdr_image_and_tonemap,
    latest_iteration_ply,
    latest_checkpoint_pth,
    iteration_ply,
    first_existing,
)

# torch.autograd.set_detect_anomaly(True)

def main():
	# =================================================================================
	# 1. 参数解析
	# =================================================================================
	args, model_params, pipeline_params = get_attack_args()

	# =================================================================================
	# 2. 环境与路径设置
	# =================================================================================
	device = torch.device(args.device)
	save_dir = Path(args.save_dir)
	# Resolve relative save_dir to repository directory to avoid cwd permission issues
	if not save_dir.is_absolute():
		repo_dir = Path(__file__).resolve().parent
		save_dir = repo_dir / save_dir

	# New: Create a timestamped subdirectory for all outputs using Beijing Time (UTC+8)
	beijing_tz = timezone(timedelta(hours=8))
	timestamp = datetime.now(beijing_tz).strftime("%m%d_%H%M%S") + "_Beijing"
	save_dir = save_dir / timestamp

	# If a file exists with the same name, redirect to a new folder name
	if save_dir.exists() and not save_dir.is_dir():
		save_dir = save_dir.with_name(save_dir.name + "_dir")
	# Try mkdir with graceful fallbacks
	try:
		save_dir.mkdir(parents=True, exist_ok=True)
	except PermissionError:
		fallback1 = Path('/workspace/RGA') / save_dir.name
		try:
			fallback1.mkdir(parents=True, exist_ok=True)
			save_dir = fallback1
		except PermissionError as e:
			from time import time as _now
			fallback2 = Path('/workspace/RGA') / f"{save_dir.name}_{int(_now())}"
			try:
				fallback2.mkdir(parents=True, exist_ok=True)
				save_dir = fallback2
			except PermissionError as e2:
				raise PermissionError(
					f"Failed to create save dir at '{save_dir}' and '{fallback1}'. "
					"Please set --save_dir to a writable path under /workspace."
				) from e2
	print(f"[消息] 输出将保存到: {save_dir}")

	# Save command line arguments to a file
	args_save_path = save_dir / "args.txt"
	with open(args_save_path, 'w') as f:
		for k, v in sorted(vars(args).items()):
			f.write(f"{k}: {v}\n")
	print(f"[消息] 命令行参数已保存到: {args_save_path}")

	# =================================================================================
	# 3. 加载场景与高斯模型
	# =================================================================================
	dataset = model_params.extract(args)
	# dataset.eval = True # Reverted: We will manually split cameras instead.

	if args.environment_texture != "":
		gaussians = GaussianModel(dataset.sh_degree, environment_texture=args.environment_texture, environment_scale=args.environment_scale)
	else:
		gaussians = GaussianModel(dataset.sh_degree)

	if args.environment_texture == "":
		print("[消息] 未提供 environment_texture，将尝试从最新的检查点加载 'envlight'...")
		model_dir = Path(dataset.model_path)
		latest_ckpt_path = latest_checkpoint_pth(model_dir)
		if latest_ckpt_path:
			print(f"[消息] 找到最新检查点: {latest_ckpt_path}")
			try:
				# The .pth file contains a tuple: (captured_model_tuple, iteration_number)
				ckpt_data_tuple, _ = torch.load(str(latest_ckpt_path), map_location=device)

				# Find the envlight state_dict by searching for a dictionary that is NOT the optimizer state.
				# This is more robust than assuming a fixed index.
				envlight_state_dict = None
				for item in ckpt_data_tuple:
					if isinstance(item, dict):
						# A simple heuristic to distinguish optimizer state_dict from others.
						# Optimizer state has 'state' and 'param_groups' keys.
						if 'state' in item and 'param_groups' in item:
							continue
						else:
							envlight_state_dict = item
							break # Found it

				if envlight_state_dict is not None:
					print("[消息] 正在从检查点加载 'env_light'...")
					# Load the state into the default envlight object that comes with the gaussians model
					gaussians.envlight.load_state_dict(envlight_state_dict)
					print("[消息] 成功加载 'envlight'。")

					# --- Start: Verification Save ---
					try:
						print("[消息] 正在保存加载的 envlight 到图片以供验证...")
						# Rebuild the base texture from the loaded state before saving
						hdr_image = cubemap_to_latlong(gaussians.envlight.base.detach(), [512, 1024]).permute(2,0,1).contiguous()
						verify_path = save_dir / 'loaded_envlight_from_ckpt.png'
						torchvision.utils.save_image(hdr_image.cpu(), str(verify_path))
						print(f"[消息] 验证图片已保存到: {verify_path}")
					except Exception as e:
						print(f"[消息] 保存验证图片失败: {e}")
					# --- End: Verification Save ---
				else:
					print("[消息] 在检查点中未找到 'env_light' 的 state_dict，将在无环境光情况下继续。")
			except Exception as e:
				print(f"[消息] 从检查点加载 envlight 失败: {e}")
		else:
			print("[消息] 未找到 '.pth' 检查点文件，将在无环境光情况下继续。")

	scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
	gaussians.get_diffuse_occ()

	# Manually clone the gaussians model once before training to avoid deepcopy issues
	gaussians_original = GaussianModel(dataset.sh_degree)
	gaussians_original.active_sh_degree = gaussians.active_sh_degree
	gaussians_original._xyz = torch.nn.Parameter(gaussians._xyz.clone().detach())
	gaussians_original._features_dc = torch.nn.Parameter(gaussians._features_dc.clone().detach())
	gaussians_original._features_rest = torch.nn.Parameter(gaussians._features_rest.clone().detach())
	gaussians_original._scaling = torch.nn.Parameter(gaussians._scaling.clone().detach())
	gaussians_original._rotation = torch.nn.Parameter(gaussians._rotation.clone().detach())
	gaussians_original._opacity = torch.nn.Parameter(gaussians._opacity.clone().detach())
	gaussians_original._albedo_init = torch.nn.Parameter(gaussians._albedo_init.clone().detach())
	if hasattr(gaussians, '_metallic_init'):
		gaussians_original._metallic_init = torch.nn.Parameter(gaussians._metallic_init.clone().detach())
	if hasattr(gaussians, '_roughness_init'):
		gaussians_original._roughness_init = torch.nn.Parameter(gaussians._roughness_init.clone().detach())
	gaussians_original.envlight.load_state_dict(gaussians.envlight.state_dict())
	gaussians_original.max_radii2D = gaussians.max_radii2D.clone().detach()
	gaussians_original.diffuse_occ = gaussians.diffuse_occ.clone().detach()
	gaussians_original.diffuse_direction_samples = gaussians.diffuse_direction_samples.clone().detach()
	if hasattr(gaussians, 'min_pts') and gaussians.min_pts is not None:
		gaussians_original.min_pts = gaussians.min_pts.clone().detach()
	if hasattr(gaussians, 'max_pts') and gaussians.max_pts is not None:
		gaussians_original.max_pts = gaussians.max_pts.clone().detach()
	gaussians_original.get_diffuse_occ()
	
	# --- Perturb Initial Albedo if Enabled ---
	if args.perturb_albedo:
		print("[消息] [扰动] 启用反照率随机初始化...")
		with torch.no_grad():
			albedo = gaussians.get_albedo_init
			min_val, max_val = albedo.min(), albedo.max()
			original_mean = albedo.mean().item()
			
			if args.albedo_init_method == 'perturb':
				print(f"[消息] [扰动] 使用 'perturb' 方法进行初始化...")
				budget = (max_val - min_val) * args.perturb_budget_factor
				# 生成初始化扰动 [-budget, budget] 并添加在原始albedo上
				perturbation = (torch.rand_like(albedo) * 2 - 1) * budget
				gaussians._albedo_init.data += perturbation

			elif args.albedo_init_method == 'random':
				print(f"[消息] [扰动] 使用 'random' 方法进行初始化...")
				# 在 [min_val, max_val] 范围内完全随机初始化
				random_values = min_val + (max_val - min_val) * torch.rand_like(albedo)
				gaussians._albedo_init.data = random_values

			perturbed_mean = gaussians.get_albedo_init.mean().item()

			print(f"[消息] [扰动] 原始反照率范围: [{min_val:.4f}, {max_val:.4f}]")
			print(f"[消息] [扰动] 扰动前平均反照率: {original_mean:.4f}")
			print(f"[消息] [扰动] 扰动后平均反照率: {perturbed_mean:.4f}")

	# DEBUG: print number of gaussians (ellipsoids)
	try:
		num_gaussians = int(gaussians.get_xyz.shape[0])
		print(f"[消息] [模型] 高斯数量={num_gaussians}")
		# opacity / albedo quick stats
		op = gaussians.get_opacity
		print(f"[消息] [模型] 不透明度统计: 最小={float(op.min().item()):.6f}, 最大={float(op.max().item()):.6f}, 平均={float(op.mean().item()):.6f}")
		alb = gaussians.get_albedo_init
		print(f"[消息] [模型] 反照率统计: 最小={float(alb.min().item()):.6f}, 最大={float(alb.max().item()):.6f}, 平均={float(alb.mean().item()):.6f}")
	except Exception as e:
		print(f"[消息] [模型] 读取高斯数量失败: {e}")

	# =================================================================================
	# 4. 初始化渲染器与优化器
	# =================================================================================
	bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
	bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
	pipe = pipeline_params.extract(args)

	# Dynamically create the optimizer based on user's choice
	if args.optimizer == 'adam':
		optimizer_min = torch.optim.Adam([gaussians._albedo_init], lr=args.lr)
		print(f"[消息] [优化器] 使用 Adam, 学习率: {args.lr}")
	elif args.optimizer == 'sgd':
		optimizer_min = torch.optim.SGD([gaussians._albedo_init], lr=args.lr, momentum=args.momentum)
		print(f"[消息] [优化器] 使用 SGD, 学习率: {args.lr}, 动量: {args.momentum}")
	elif args.optimizer == 'adamw':
		optimizer_min = torch.optim.AdamW([gaussians._albedo_init], lr=args.lr)
		print(f"[消息] [优化器] 使用 AdamW, 学习率: {args.lr}")
	else:
		# This case should not be reached due to 'choices' in argparse
		raise ValueError(f"未知的优化器类型: {args.optimizer}")

	# --- (Optional) Initialize LR Scheduler for SGD ---
	scheduler = None
	if args.optimizer == 'sgd' and args.use_lr_scheduler:
		if args.lr_scheduler_type == 'cosine':
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_min, T_max=args.epochs)
			print(f"[消息] [调度器] 已为 SGD 启用 CosineAnnealingLR 调度器, T_max={args.epochs}")
		elif args.lr_scheduler_type == 'step':
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer_min, step_size=args.lr_step_size, gamma=args.lr_gamma)
			print(f"[消息] [调度器] 已为 SGD 启用 StepLR 调度器, step_size={args.lr_step_size}, gamma={args.lr_gamma}")


	# =================================================================================
	# 5. 初始化目标检测器与损失函数
	# =================================================================================
	print("[消息] 正在初始化检测器...")

	base_path = Path('/workspace/RGA/mmdet_files')
	
	# detector_paths removed here, using global DETECTOR_PATHS

	selected_detector = DETECTOR_PATHS.get(args.detector)

	if selected_detector is None:
		raise ValueError(f"Detector '{args.detector}' not found in hardcoded paths. Available: {list(DETECTOR_PATHS.keys())}")

	cfg_path = base_path / selected_detector['config']
	ckpt_path = base_path / selected_detector['ckpt']
	print(f"[消息] 配置文件路径: {cfg_path}")
	print(f"[消息] 权重文件路径: {ckpt_path}")
	if not cfg_path.is_file():
		raise FileNotFoundError(f"Config file not found: {cfg_path}. Please check the path and ensure 'mmdet_files' is set up correctly.")
	if not ckpt_path.is_file():
		raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}. Please check the path and ensure 'mmdet_files' is set up correctly.")

	cfg_path, ckpt_path = str(cfg_path), str(ckpt_path)

	args.detector_cfg = cfg_path
	args.detector_ckpt = ckpt_path
	print(f"[消息] 配置文件路径: {cfg_path}")
	print(f"[消息] 权重文件路径: {ckpt_path}")
	# MMDetLoss is a wrapper, but for custom loss logic, we use the model directly.
	detector = init_detector(cfg_path, ckpt_path, device=device)
	for p in detector.parameters():
		p.requires_grad = False
	detector.eval()
	if not hasattr(detector, 'CLASSES'):
		detector.CLASSES = coco_classes
	print("[消息] 检测器初始化完成。")

	# =================================================================================
	# 5.5. (Optional) Initialize LBM Relighter
	# =================================================================================
	relighter = None
	if args.enable_lbm_relight:
		print("[消息] LBM 背景重打光已启用。正在初始化 LBMRelighter...")
		try:
			relighter = LBMRelighter(ckpt_dir=args.lbm_ckpt_dir, device=device)
			print("[消息] LBMRelighter 初始化完成。")
		except Exception as e:
			print(f"[警告] LBMRelighter 初始化失败: {e}。将禁用背景重打光。")
			relighter = None


	# =================================================================================
	# 6. 开始训练循环
	# =================================================================================
	global_step = 60000
	batch_size = args.batch_size
	
	# New: Initialize TrainingLogger
	logger = TrainingLogger(save_dir)

	# --- Manual Train/Test Split ---
	print("[消息] 正在手动划分训练集与测试集...")
	all_cameras = scene.getTrainCameras() # With eval=False, this gets all cameras
	random.shuffle(all_cameras)
	split_idx = int(len(all_cameras) * 0.9)
	train_cameras = all_cameras[:split_idx]
	test_cameras = all_cameras[split_idx:]

	if args.max_cams > 0:
		train_cameras = train_cameras[:args.max_cams]
		test_cameras = test_cameras[:args.max_cams]

	print(f"[消息] 划分完成. 训练集: {len(train_cameras)} 张, 测试集: {len(test_cameras)} 张")

	for epoch in range(1, args.epochs + 1):
		print(f"[消息] [第 {epoch}/{args.epochs} 輪] 开始")
		
		cams_to_process = train_cameras
		if args.max_cams > 0:
			cams_to_process = cams_to_process[:args.max_cams]
		
		# Create batches from camera list
		cam_batches = [cams_to_process[i:i + batch_size] for i in range(0, len(cams_to_process), batch_size)]
		
		pbar = tqdm(enumerate(cam_batches), total=len(cam_batches), desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

		batch_total_loss_for_display = 0.0

		for batch_idx, cam_batch in pbar:

			# --- Forward Pass and Loss Calculation ---
			is_first_batch = pbar.n == 0
			batch_total_loss, batch_cls_loss, batch_reg_loss, detect_imgs, vis_data = compute_batch_loss(
				cam_batch, gaussians, pipe, bg, global_step, args, dataset, 
				gaussians_original, relighter, detector, save_dir, epoch, batch_idx
			)
			print(f"batch_total_loss: {batch_total_loss.item()}")
			print(f"batch_cls_loss: {batch_cls_loss.item()}")
			print(f"batch_reg_loss: {batch_reg_loss.item()}")
			if batch_total_loss is not None:
				batch_total_loss_for_display = batch_total_loss.item()
			else:
				batch_total_loss_for_display = 0.0

			# New visualization logic for the first batch of each epoch
			if is_first_batch and vis_data:
				vis_save_dir = save_dir / 'visualizations'
				vis_save_dir.mkdir(parents=True, exist_ok=True)
				for i, vis_item in enumerate(vis_data):
					# Ensure camera name exists for the item
					if i < len(cam_batch):
						cam_name = cam_batch[i].image_name
						save_path = vis_save_dir / f'epoch_{epoch:03d}_{cam_name}.png'
						save_visualization_grid(save_path, vis_item)

			if batch_total_loss is None: # Skip if batch was empty
				continue

			# --- Standard Optimization Step ---
			optimizer_min.zero_grad()
			batch_total_loss.backward()
			logger.log_iteration(batch_total_loss.item(), batch_cls_loss.item(), batch_reg_loss.item())
			# optimizer_min.step()
			
			# --- 调试: 检查 albedo 是否变化 ---
			try:
				if gaussians._albedo_init.grad is not None:
					grad_norm = gaussians._albedo_init.grad.norm().item()
					print(f"    [调试] Albedo Grad Norm: {grad_norm:.6f}")
				else:
					print("    [调试] Albedo gradient is None.")
				
				albedo_before_step = gaussians._albedo_init.data.clone()
				optimizer_min.step()
				albedo_after_step = gaussians._albedo_init.data

				total_diff = torch.sum(torch.abs(albedo_after_step - albedo_before_step)).item()
				print(f"    [调试] Albedo 差值绝对值总和: {total_diff:.6f}")

				mean_before = albedo_before_step.mean().item()
				mean_after = albedo_after_step.mean().item()
				print(f"    [调试] Albedo Mean: {mean_before:.6f} -> {mean_after:.6f}")
				
				if total_diff < 1e-9:
					print("    [警告] Albedo 值在优化步骤后未发生变化。")
				else:
					print("    [消息] Albedo 值已成功更新。")
			except Exception as e:
				print(f"    [错误] 在 albedo 调试代码块中出错: {e}")
				optimizer_min.step() # 如果调试代码失败，仍确保执行优化
			# --- 调试结束 ---


		# 5. Logging and saving
		avg_loss_display = batch_total_loss_for_display / len(cam_batch) if len(cam_batch) > 0 else 0
		pbar.set_postfix(batch_loss=f"{avg_loss_display:.4f}")

		# --- Step the LR Scheduler at the end of the epoch ---
		if scheduler is not None:
			scheduler.step()
			current_lr = scheduler.get_last_lr()[0]
			print(f"\n[消息] [第 {epoch}/{args.epochs} 輪] LR Scheduler 已更新. 当前学习率: {current_lr:.6f}")


		global_step += len([c for c in cam_batch if (Path(dataset.source_path) / 'annos' / f'{c.image_name}.json').exists()])

		# save ply per-epoch
		if gaussians is not None:
			gaussians.save_ply(str(save_dir / f'point_cloud_epoch_{epoch:03d}.ply'))
			print(f"[消息] [第 {epoch}/{args.epochs} 輪] 已保存点云: point_cloud_epoch_{epoch:03d}.ply")

		# --- Evaluation Phase ---
		if test_cameras:
			# --- Evaluate on Test Set ---
			eval_cams_test = test_cameras
			if args.max_cams > 0:
				eval_cams_test = test_cameras[:args.max_cams]
			
			asr_test, succ_test, total_test, ap50_test = evaluate(
				eval_cams_test, gaussians, pipe, bg, args, dataset, 
				gaussians_original, relighter, detector, epoch, save_dir, 'test'
			)

			asr_train = 0.0 # Default value
			if args.eval_on_train:
				# --- Evaluate on Train Set ---
				eval_cams_train = train_cameras
				if args.max_cams > 0:
					# To save time, we can evaluate on a subset of the training data
					eval_cams_train = train_cameras[:args.max_cams]
				
				asr_train, _, _, _ = evaluate(
					eval_cams_train, gaussians, pipe, bg, args, dataset, 
					gaussians_original, relighter, detector, epoch, save_dir, 'train'
				)

			logger.log_epoch(epoch, asr_test, asr_train, ap50_test)


	# --- After all epochs, plot and save ASR curve ---
	logger.plot_iteration_losses()
	logger.plot_epoch_losses()
	logger.plot_asr_and_loss()
	logger.plot_ap_curve()

	# =================================================================================
	# 7. 最终评估与记录 (可选) - 修改为遍历所有检测器
	# =================================================================================
	if args.run_final_eval:
		print("\n[消息] 所有训练轮次完成。开始多检测器最终评估...")
		
		# Define datasets
		full_cameras = train_cameras + test_cameras
		
		# --- STAGE 1: Render all required images once ---
		final_test_img_dir = render_and_save_final_images(
			test_cameras, gaussians, pipe, bg, args, dataset, 
			gaussians_original, relighter, save_dir, 'test'
		)
		final_full_img_dir = render_and_save_final_images(
			full_cameras, gaussians, pipe, bg, args, dataset, 
			gaussians_original, relighter, save_dir, 'full'
		)

		# --- STAGE 2: Evaluate on saved images with all detectors ---
		log_file_path = save_dir / 'training_log.txt'
		with open(log_file_path, 'a', encoding='utf-8') as f:
			f.write("\n\n==================================================\n")
			f.write(f"=== Final Cross-Detector Evaluation Results (Epoch {args.epochs}) ===\n")
			f.write("==================================================\n")
			beijing_tz = timezone(timedelta(hours=8))
			f.write(f"Timestamp: {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

		# Release the training detector
		del detector
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			
		base_path = Path('/workspace/RGA/mmdet_files')
		
		for det_name in DETECTOR_PATHS.keys():
			print(f"\n>>> [评估] 正在加载检测器: {det_name} ...")
			
			try:
				# Init detector
				cfg_path = str(base_path / DETECTOR_PATHS[det_name]['config'])
				ckpt_path = str(base_path / DETECTOR_PATHS[det_name]['ckpt'])
				
				curr_detector = init_detector(cfg_path, ckpt_path, device=device)
				if not hasattr(curr_detector, 'CLASSES'):
					curr_detector.CLASSES = coco_classes
				
				# 1. Evaluate on Test Set (Group 1)
				asr_test, succ_test, total_test, ap50_test = evaluate_from_saved_images(
					curr_detector, final_test_img_dir, Path(dataset.source_path) / 'annos', args
				)
				
				# 2. Evaluate on Full Set (Group 2)
				asr_full, succ_full, total_full, ap50_full = evaluate_from_saved_images(
					curr_detector, final_full_img_dir, Path(dataset.source_path) / 'annos', args
				)
				
				# Log results
				print(f"  - [结果] 检测器: {det_name}, 测试集 ASR: {asr_test:.4f}, 全集 ASR: {asr_full:.4f}")
				with open(log_file_path, 'a', encoding='utf-8') as f:
					f.write(f"Detector: {det_name}\n")
					f.write(f"  - [Test Set] ASR: {asr_test:.4f} ({succ_test}/{total_test}), AP@0.5: {ap50_test:.4f}\n")
					f.write(f"  - [Full Set] ASR: {asr_full:.4f} ({succ_full}/{total_full}), AP@0.5: {ap50_full:.4f}\n")
					f.write("-" * 30 + "\n")
					
				# Cleanup
				del curr_detector
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					
			except Exception as e:
				print(f"[错误] 评估检测器 {det_name} 时发生错误: {e}")
				with open(log_file_path, 'a', encoding='utf-8') as f:
					f.write(f"Detector: {det_name} - FAILED: {e}\n")
					f.write("-" * 30 + "\n")
		
		print(f"\n[消息] 最终多检测器评估完成。结果已保存到: {log_file_path}")

	else:
		print("\n[消息] 根据设置，已跳过最终评估步骤。")


	# =================================================================================
	# 8. 保存最终模型
	# =================================================================================
	if gaussians is not None:
		gaussians.save_ply(str(save_dir / 'point_cloud_final.ply'))


if __name__ == '__main__':
	main()
