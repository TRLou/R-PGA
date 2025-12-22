from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import subprocess
import torch
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
import torchvision
from envlight.utils import cubemap_to_latlong
from utils.main_utils import coco_classes
from mmdet.apis import init_detector
from lbm_relit import LBMRelighter
from utils.log_utils import TrainingLogger
from attack_options import get_attack_args
from submodules.envlight.envlight.light import EnvLight as EnvLightClass

from train_func_hdr import (
    DETECTOR_PATHS,
    save_visualization_grid,
    compute_batch_loss,
    evaluate,
    render_and_save_final_images_mw,
    evaluate_from_saved_images,
    visualize_hdr_bank_from_dir,
    visualize_hdr_bases_with_random_views,
    load_hdr_image_and_tonemap,
    latest_iteration_ply,
    latest_checkpoint_pth,
    iteration_ply,
    first_existing,
)

# torch.autograd.set_detect_anomaly(True)

def compute_sh9_rgb_from_latlong(env_img: np.ndarray) -> np.ndarray:
    """
    Computes 2nd order (9 bands) spherical harmonics coefficients for each of the R,G,B channels,
    resulting in a 27-dimensional vector. This function is ported from the data preparation script
    to ensure consistency between training and inference.
    Assumes the input is a lat-long projection where rows correspond to theta in [0, pi]
    and columns to phi in [0, 2*pi).
    """
    assert env_img.ndim == 3 and env_img.shape[-1] == 3, "env_img must be HxWx3"
    H, W, _ = env_img.shape
    # Grid setup
    theta = np.linspace(0.0, np.pi, H, endpoint=False) + (np.pi / H) * 0.5
    phi = np.linspace(0.0, 2 * np.pi, W, endpoint=False) + (2 * np.pi / W) * 0.5
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")  # HxW
    # Direction vectors
    sin_t = np.sin(theta_grid)
    cos_t = np.cos(theta_grid)
    cos_p = np.cos(phi_grid)
    sin_p = np.sin(phi_grid)
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    # 9 SH basis functions
    Y00 = 0.28209479177387814 * np.ones_like(x)
    Y1m1 = 0.4886025119029199 * y
    Y10 = 0.4886025119029199 * z
    Y11 = 0.4886025119029199 * x
    Y2m2 = 1.0925484305920792 * x * y
    Y2m1 = 1.0925484305920792 * y * z
    Y20 = 0.31539156525252005 * (3.0 * z * z - 1.0)
    Y21 = 1.0925484305920792 * x * z
    Y22 = 0.5462742152960396 * (x * x - y * y)
    bases = [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22]
    # Solid angle weights
    dtheta = np.pi / H
    dphi = 2.0 * np.pi / W
    domega = sin_t * dtheta * dphi  # HxW
    # Integrate over each channel
    L = env_img.astype(np.float32)  # HxWx3
    coefs = []
    for c in range(3):
        Lc = L[..., c]
        for B in bases:
            coefs.append(np.sum(Lc * B * domega, dtype=np.float64))
    coefs = np.array(coefs, dtype=np.float32)  # 27
    return coefs


def base_cubemap_to_sh(base_cubemap: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Converts an envlight base cubemap tensor into a Spherical Harmonics tensor,
    aligning with the LBM model's training data pipeline.
    """
    if base_cubemap is None:
        return torch.zeros(27, dtype=torch.float32, device=device)

    # 1. Convert base cubemap to latlong. The input `base_cubemap` is already a tensor.
    # The 'res' argument is required by this version of cubemap_to_latlong.
    latlong_t = cubemap_to_latlong(base_cubemap.to(device), res=(64, 128))
    latlong_np = latlong_t.detach().cpu().numpy().astype(np.float32)

    # 2. Resize latlong to fixed small resolution (128, 64)
    target_w, target_h = 128, 64
    # 修正：使用百分位数进行归一化，以抵抗离群值
    # max_val = np.percentile(latlong_np, 99.5) if np.max(latlong_np) > 0 else 1.0
    max_val = np.max(latlong_np) if np.max(latlong_np) > 0 else 1.0
    latlong_norm = np.clip(latlong_np / max_val, 0.0, 1.0)
    latlong_u8 = (latlong_norm * 255.0).astype(np.uint8)
    im = Image.fromarray(latlong_u8)
    im_resized = im.resize((target_w, target_h), resample=Image.BILINEAR)
    resized_u8 = np.asarray(im_resized).astype(np.float32) / 255.0
    latlong_resized = resized_u8 * max_val

    # 3. Compute SH coefficients
    sh_coeffs_np = compute_sh9_rgb_from_latlong(latlong_resized)

    # 4. Convert to tensor and return
    return torch.from_numpy(sh_coeffs_np).to(device)


class EnvLightReplayBuffer:
	"""
	用于存储与复用 envlight 的“基底立方体贴图”（base）。
	仅在 CPU 上存储张量的克隆，避免显存占用与泄漏。
	"""
	def __init__(self, max_size: int = 100, initial_states=None, replace_strategy: str = "fifo"):
		self.max_size = int(max_size)
		self.replace_strategy = str(replace_strategy).lower().strip() if replace_strategy is not None else "fifo"
		self.storage = []
		# Track which entry was sampled last (for replace_self policy)
		self.last_sampled_idx = None
		if initial_states:
			for st in initial_states:
				self.push(st)

	def _to_cpu_clone(self, base_tensor: torch.Tensor):
		return base_tensor.detach().cpu().clone()

	def push(self, base_tensor: torch.Tensor):
		if self.max_size <= 0:
			return
		cpu_copy = self._to_cpu_clone(base_tensor)
		# Policy A: replace the entry that was sampled for the current max-phase
		if self.replace_strategy in ("replace_self", "replace-sampled", "replace_sampled"):
			if self.storage and self.last_sampled_idx is not None:
				try:
					idx = int(self.last_sampled_idx)
				except Exception:
					idx = None
				if idx is not None and 0 <= idx < len(self.storage):
					self.storage[idx] = cpu_copy
					return
			# If we can't replace (no sampled idx), fall back to append/FIFO behavior below.

		# Policy B (default): FIFO replacement when full
		if len(self.storage) >= self.max_size:
			self.storage.pop(0)
		self.storage.append(cpu_copy)

	def sample(self):
		if not self.storage:
			self.last_sampled_idx = None
			return None
		idx = random.randrange(len(self.storage))
		self.last_sampled_idx = idx
		# 返回深拷贝，避免后续修改影响 Buffer 内容
		return deepcopy(self.storage[idx])

def sgld_step(module: torch.nn.Module, lr: float, noise_std: float):
	"""
	对给定 module 的所有参数执行一次 SGLD（梯度上升）更新：
		param = param + lr * grad + N(0, noise_std)
	要求在调用前已执行 backward() 得到 grad。
	"""
	with torch.no_grad():
		for p in module.parameters():
			if p.grad is None:
				continue
			# 梯度上升 + 高斯噪声
			p.add_(lr * p.grad)
			if noise_std > 0.0:
				p.add_(torch.randn_like(p) * noise_std)

def main():
	# =================================================================================
	# 1. 参数解析
	# =================================================================================
	args, model_params, pipeline_params = get_attack_args()

	# --- START DEBUG OVERRIDE ---
	# print("\n" + "="*50)
	# print("[调试] 正在覆盖参数以隔离HDR源...")
	# args.use_replay_buffer = False
	# args.hdr_bank_dir = '' # disable hdr bank
	# print("[调试] - use_replay_buffer 设置为 False")
	# print("[调试] - hdr_bank_dir 设置为空")
	# print("="*50 + "\n")
	# --- END DEBUG OVERRIDE ---

	# 记录初始（从 checkpoint 或初始化得到的）envlight.base 的 CPU 拷贝
	initial_env_base_cpu = None
	# 回退默认参数，以避免修改 get_attack_args 的实现
	if not hasattr(args, 'use_replay_buffer'):
		args.use_replay_buffer = False
	if not hasattr(args, 'buffer_size'):
		args.buffer_size = 100
	if not hasattr(args, 'buffer_replace_strategy'):
		args.buffer_replace_strategy = 'fifo'

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
					# 加载后重建 base 与 mips，确保采样链与参数一致
					try:
						gaussians.envlight.build_base()
						gaussians.envlight.build_mips()
						print("[消息] 已根据加载参数重建 envlight base 与 mips。")
						# 保存初始 base（CPU 拷贝）
						try:
							initial_env_base_cpu = gaussians.envlight.base.detach().cpu().clone()
						except Exception:
							initial_env_base_cpu = None
					except Exception as e:
						print(f"[警告] 重建 envlight base/mips 失败: {e}")

				else:
					print("[消息] 在检查点中未找到 'env_light' 的 state_dict，将在无环境光情况下继续。")
			except Exception as e:
				print(f"[消息] 从检查点加载 envlight 失败: {e}")
		else:
			print("[消息] 未找到 '.pth' 检查点文件，将在无环境光情况下继续。")

	# 若未能从 checkpoint 设置初始 base，则以当前 base 为初始备份
	if initial_env_base_cpu is None:
		try:
			initial_env_base_cpu = gaussians.envlight.base.detach().cpu().clone()
		except Exception:
			initial_env_base_cpu = None

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
	# 同步一次 envlight.base，state_dict 不包含非参数的 base
	with torch.no_grad():
		try:
			gaussians_original.envlight.base = gaussians.envlight.base.detach().clone()
		except Exception:
			pass
	gaussians_original.max_radii2D = gaussians.max_radii2D.clone().detach()
	gaussians_original.diffuse_occ = gaussians.diffuse_occ.clone().detach()
	gaussians_original.diffuse_direction_samples = gaussians.diffuse_direction_samples.clone().detach()
	if hasattr(gaussians, 'min_pts') and gaussians.min_pts is not None:
		gaussians_original.min_pts = gaussians.min_pts.clone().detach()
	if hasattr(gaussians, 'max_pts') and gaussians.max_pts is not None:
		gaussians_original.max_pts = gaussians.max_pts.clone().detach()
	gaussians_original.get_diffuse_occ()
	
	with torch.no_grad():
		initial_raw_albedo = gaussians._albedo_init.data
		print(f"[消息] [检查] 模型加载后, _albedo_init 范围: [{initial_raw_albedo.min().item():.4f}, {initial_raw_albedo.max().item():.4f}]")
		gaussians._albedo_init.data.zero_()
		zeroed_raw_albedo = gaussians._albedo_init.data
		print(f"[消息] [修正] _albedo_init 已全部设置为 0. 新范围: [{zeroed_raw_albedo.min().item():.4f}, {zeroed_raw_albedo.max().item():.4f}]")

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
				# gaussians._albedo_init.data.zero_()

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

	# 最大化阶段优化器（仅 envlight 参数）
	optimizer_max = None
	if getattr(args, 'enable_min_max', False):
		# 仅优化 base_train，相当于直接在 base 上作加性更新；冻结 net/init_base
		for p in gaussians.envlight.net.parameters():
			p.requires_grad = False
		if hasattr(gaussians.envlight, 'init_base'):
			gaussians.envlight.init_base.requires_grad_(False)
		optimizer_max = torch.optim.Adam([gaussians.envlight.base_train], lr=args.env_lr)
		print(f"[消息] [优化器-最大化] 使用 Adam 优化 envlight.base_train, 学习率: {args.env_lr}")

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


	# 如果提供了 HDR 银行目录，预载入 base
	initial_bases = []
	hdr_bank_dir = getattr(args, 'hdr_bank_dir', '')
	if isinstance(hdr_bank_dir, str) and len(hdr_bank_dir) > 0:
		hdr_dir_path = Path(hdr_bank_dir)
		if hdr_dir_path.is_dir():
			hdr_files = sorted([p for p in hdr_dir_path.iterdir() if p.suffix.lower() in ['.hdr', '.exr']])
			if len(hdr_files) == 0:
				print(f"[消息] [HDR Bank] 目录 '{hdr_bank_dir}' 中未找到 .hdr/.exr 文件，跳过预加载。")
			else:
				print(f"[消息] [HDR Bank] 在 '{hdr_bank_dir}' 发现 {len(hdr_files)} 个 HDR/EXR 文件，开始预加载 base...")
				for fp in hdr_files:
					try:
						device_for_loading = 'cuda' if torch.cuda.is_available() else 'cpu'
						tmp_env = EnvLightClass(
							path=str(fp),
							device=device_for_loading,
							scale=args.environment_scale,
							min_res=16,
							max_res=512,
							trainable=False
						)
						# 直接缓存 base（立方体贴图），用于后续切换，不进行 build_base 覆盖
						initial_bases.append(tmp_env.base.detach().cpu().clone())
						print(f"[消息] [HDR Bank] 已载入 base: {fp.name}")
					except Exception as e:
						print(f"[警告] [HDR Bank] 载入失败: {fp.name}: {e}")
					finally:
						try:
							del tmp_env
						except Exception:
							pass
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
		else:
			print(f"[消息] [HDR Bank] '{hdr_bank_dir}' 非目录或不存在，跳过预加载。")

	# 初始化经验重放缓冲区（可选）
	replay_buffer = None
	if getattr(args, 'use_replay_buffer', False):
		# When using a buffer with min-max, we initialize it only with clean states from the HDR bank.
		# The base from the checkpoint is deliberately not added to avoid starting with a potentially biased state.
		initial_states = list(initial_bases) if initial_bases else []
		replay_buffer = EnvLightReplayBuffer(
			max_size=int(getattr(args, 'buffer_size', 100)),
			initial_states=initial_states,
			replace_strategy=str(getattr(args, 'buffer_replace_strategy', 'fifo'))
		)

	# --- 训练前从 Replay Buffer 采样一个干净的 base ---
	if getattr(args, 'use_replay_buffer', False) and replay_buffer is not None and replay_buffer.storage:
		try:
			print("[消息] [预处理] 尝试从 Replay Buffer 中采样一个初始 base...")
			initial_sampled_base = replay_buffer.sample()
			if initial_sampled_base is not None:
				gaussians.envlight.base = initial_sampled_base.to(device)
				gaussians.envlight.build_mips()
				print("[消息] [预处理] 成功！训练将从一个干净的、采样的 base 开始。")
			else:
				print("[警告] [预处理] Replay Buffer 为空或采样失败，将使用来自 ckpt 的 base。")
		except Exception as e:
			print(f"[警告] [预处理] 采样初始 base 失败: {e}")


	# --- 预可视化：遍历 HDR 目录，随机抽取若干视角渲染 ---
	try:
		if getattr(args, 'enable_hdr_bank_vis_pre', False):
			hdr_bank_dir = getattr(args, 'hdr_bank_dir', '')
			if isinstance(hdr_bank_dir, str) and len(hdr_bank_dir) > 0:
				views_for_vis = scene.getTrainCameras()
				num_views = int(getattr(args, 'hdr_vis_num_views', 5))
				seed = int(getattr(args, 'hdr_vis_seed', 0))
				save_root = save_dir / 'hdrbank_vis_pre'
				visualize_hdr_bank_from_dir(
					cameras=views_for_vis,
					gaussians=gaussians,
					pipe=pipe,
					bg=bg,
					args=args,
					dataset=dataset,
					hdr_bank_dir=Path(hdr_bank_dir),
					save_root=save_root,
					num_views=num_views,
					seed=seed
				)
	except Exception as e:
		print(f"[警告] [HDR-VIS PRE] 可视化失败: {e}")

	for epoch in range(1, args.epochs + 1):
		print(f"[消息] [第 {epoch}/{args.epochs} 輪] 开始")
		
		# 每个 epoch 开始：可选打乱视角顺序，避免固定分配到 min/max
		if getattr(args, 'shuffle_each_epoch', True):
			random.shuffle(train_cameras)
			random.shuffle(test_cameras)
			print("[消息] [Epoch] 已打乱训练/测试视角顺序。")

		
		cams_to_process = train_cameras
		if args.max_cams > 0:
			cams_to_process = cams_to_process[:args.max_cams]
		
		# Create batches from camera list
		cam_batches = [cams_to_process[i:i + batch_size] for i in range(0, len(cams_to_process), batch_size)]
		
		pbar = tqdm(enumerate(cam_batches), total=len(cam_batches), desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

		batch_total_loss_for_display = 0.0
		prev_phase = None

		for batch_idx, cam_batch in pbar:

			# --- 先确定当前批次所处的阶段：min 或 max，并在切入 max 前进行重放采样 ---
			phase = 'min'
			if getattr(args, 'enable_min_max', False):
				total_cycle = max(1, int(args.min_steps) + int(args.max_steps))
				idx_in_cycle = batch_idx % total_cycle
				if idx_in_cycle >= int(args.min_steps):
					phase = 'max'
			pbar.set_postfix_str(f"phase={phase}")

			# 若刚从 min 切换到 max，则在前向传播之前先加载重放的 envlight 状态，避免在 backward 前修改参数导致图失效
			if phase == 'max' and prev_phase != 'max' and getattr(args, 'use_replay_buffer', False) and replay_buffer is not None:
				try:
					# Always sample (no exploration branch)
					sampled_base = replay_buffer.sample()
					if sampled_base is not None:
						gaussians.envlight.base = sampled_base.to(device=bg.device if hasattr(bg, 'device') else device)
						# 同步给 gaussians_original，确保两者共享同一环境光
						with torch.no_grad():
							try:
								gaussians_original.envlight.base = gaussians.envlight.base.detach().clone()
							except Exception:
								pass
						print("[消息] [ReplayBuffer] 已从缓冲区加载 base。")
					else:
						print("[警告] [ReplayBuffer] 采样失败（buffer 为空），将使用当前 envlight 状态。")
				except Exception as e:
					print(f"[警告] [ReplayBuffer] 加载采样状态失败: {e}")

			# --- HDR->SH Conversion ---
			# 在计算损失之前，确保当前光照的 SH 系数已计算并附加到高斯模型上
			with torch.no_grad():
				current_base = gaussians.envlight.base
				# 使用当前的 envlight base 计算 SH 系数
				sh_coeffs_tensor = base_cubemap_to_sh(current_base, device)
				# 附加到 gaussians 对象，以便在 relighter 内部访问
				gaussians.hdr_sh_coeffs = sh_coeffs_tensor


			# --- Forward Pass and Loss Calculation（在可能的重放切换之后进行） ---
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
			if bool(getattr(args, 'save_visualizations', False)) and is_first_batch and vis_data:
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

			if phase == 'min':
				# --- 最小化阶段：仅优化 albedo ---
				print("is in min phase")
				optimizer_min.zero_grad()
				batch_total_loss.backward()
				logger.log_iteration(batch_total_loss.item(), batch_cls_loss.item(), batch_reg_loss.item())
				print("requires_grad(albedo_init):", gaussians._albedo_init.requires_grad)
				print("grad(albedo_init) is None:", gaussians._albedo_init.grad is None)
				if gaussians._albedo_init.grad is not None:
					print("grad_norm(albedo_init):", gaussians._albedo_init.grad.norm().item())
				
				# （已移除 Pareto 投影相关逻辑）
				
				# --- Step & Clamp albedo ---
				try:
					albedo_before_step = gaussians._albedo_init.data.clone()
					optimizer_min.step()
					with torch.no_grad():
						gaussians._albedo_init.data.clamp_(-0.5, 0.5)
					albedo_after_step = gaussians._albedo_init.data
					total_diff = torch.sum(torch.abs(albedo_after_step - albedo_before_step)).item()
					print(f"    [调试] Albedo 差值绝对值总和: {total_diff:.6f}")
				except Exception as e:
					print(f"    [错误] 在 albedo 更新阶段出错: {e}")
					optimizer_min.step()
					with torch.no_grad():
						gaussians._albedo_init.data.clamp_(-0.5, 0.5)
			else:
				print("is in max phase")
				# --- 最大化阶段：仅优化 envlight（梯度上升） ---
				if optimizer_max is not None:
					# 记录上一步参数，用于“每次迭代”的变化量裁剪
					with torch.no_grad():
						bt_prev = gaussians.envlight.base_train.data.clone() if hasattr(gaussians.envlight, 'base_train') else None
					optimizer_max.zero_grad()
					(-batch_total_loss).backward()
					# 支持 SGLD 或 Adam：仅会更新 requires_grad=True 的参数（此处仅 base_train）
					if getattr(args, 'use_sgld', False):
						sgld_step(
							gaussians.envlight,
							lr=float(getattr(args, 'sgld_lr', 1e-2)),
							noise_std=float(getattr(args, 'sgld_noise_std', 1e-4)),
						)
					else:
						optimizer_max.step()
					# --- Clamp envlight 参数保持真实范围，并重建 mips ---
					with torch.no_grad():
						# 每次迭代的相对变化裁剪：param = prev + clamp(param - prev, [-bound, +bound])
						if hasattr(gaussians.envlight, 'base_train') and isinstance(gaussians.envlight.base_train, torch.nn.Parameter) and bt_prev is not None:
							delta_bt = gaussians.envlight.base_train.data - bt_prev
							delta_bt.clamp_(-args.env_delta_max, args.env_delta_max)
							gaussians.envlight.base_train.data = bt_prev + delta_bt
						try:
							# gaussians.envlight.build_base()
							gaussians.envlight.build_mips()
						except Exception as e:
							print(f"[警告] 构建 envlight mips 失败: {e}")
					# --- 在 Max 阶段结束后，将当前状态写入重放缓冲区 ---
					if getattr(args, 'use_replay_buffer', False) and replay_buffer is not None:
						try:
							replay_buffer.push(gaussians.envlight.base.detach())
						except Exception as e:
							print(f"[警告] [ReplayBuffer] push 当前 base 失败: {e}")

			# 更新前一阶段状态
			prev_phase = phase


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
		# if gaussians is not None:
		# 	gaussians.save_ply(str(save_dir / f'point_cloud_epoch_{epoch:03d}.ply'))
		# 	print(f"[消息] [第 {epoch}/{args.epochs} 輪] 已保存点云: point_cloud_epoch_{epoch:03d}.ply")

		# --- Evaluation Phase ---
		if test_cameras:
			# --- Evaluate on Test Set ---
			eval_cams_test = test_cameras
			if args.max_cams > 0:
				eval_cams_test = test_cameras[:args.max_cams]

			# 在评估前将 envlight.base 恢复为初始（checkpoint）版本（两套模型同步）
			prev_gauss_base_cpu = None
			prev_orig_base_cpu = None
			try:
				prev_gauss_base_cpu = gaussians.envlight.base.detach().cpu().clone()
			except Exception:
				prev_gauss_base_cpu = None
			try:
				prev_orig_base_cpu = gaussians_original.envlight.base.detach().cpu().clone()
			except Exception:
				prev_orig_base_cpu = None

			selected_base_cpu = initial_env_base_cpu if initial_env_base_cpu is not None else prev_gauss_base_cpu
			if selected_base_cpu is not None:
				with torch.no_grad():
					try:
						gaussians.envlight.base = selected_base_cpu.to(device)
						gaussians.envlight.build_mips()
					except Exception as _:
						pass
					try:
						gaussians_original.envlight.base = selected_base_cpu.to(device)
						gaussians_original.envlight.build_mips()
					except Exception as _:
						pass

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

			# 评估完成后恢复到评估前的 base（继续训练使用）
			if prev_gauss_base_cpu is not None:
				with torch.no_grad():
					try:
						gaussians.envlight.base = prev_gauss_base_cpu.to(device)
						gaussians.envlight.build_mips()
					except Exception:
						pass
			if prev_orig_base_cpu is not None:
				with torch.no_grad():
					try:
						gaussians_original.envlight.base = prev_orig_base_cpu.to(device)
						gaussians_original.envlight.build_mips()
					except Exception:
						pass

			logger.log_epoch(epoch, asr_test, asr_train, ap50_test)


	# --- After all epochs, plot and save ASR curve ---
	logger.plot_iteration_losses()
	logger.plot_epoch_losses()
	logger.plot_asr_and_loss()
	logger.plot_ap_curve()

	# --- 训练后可视化：基于 ReplayBuffer 或当前 envlight.base ---
	try:
		if getattr(args, 'enable_hdr_bank_vis_post', False):
			bases = []
			if 'replay_buffer' in locals() and replay_buffer is not None and getattr(replay_buffer, 'storage', None):
				for i, b in enumerate(replay_buffer.storage):
					bases.append((f"buf_{i:03d}", b))
			else:
				try:
					bases = [("current", gaussians.envlight.base.detach().cpu().clone())]
				except Exception:
					bases = []
			if bases:
				views_for_vis = test_cameras if test_cameras else scene.getTrainCameras()
				num_views = int(getattr(args, 'hdr_vis_num_views', 5))
				seed = int(getattr(args, 'hdr_vis_seed', 0))
				save_root = save_dir / 'hdrbank_vis_post'
				visualize_hdr_bases_with_random_views(
					cameras=views_for_vis,
					gaussians=gaussians,
					pipe=pipe,
					bg=bg,
					args=args,
					dataset=dataset,
					hdr_bases=bases,
					save_root=save_root,
					num_views=num_views,
					seed=seed
				)
	except Exception as e:
		print(f"[警告] [HDR-VIS POST] 可视化失败: {e}")

	# =================================================================================
	# 7. 最终评估与记录 (可选) - 修改为遍历所有检测器
	# =================================================================================
	if args.run_final_eval:
		print("\n[消息] 所有训练轮次完成。开始多检测器最终评估...")
		
		# Define datasets
		full_cameras = train_cameras + test_cameras

		# 在最终评估前恢复 envlight.base 为初始（checkpoint）版本（两套模型同步）
		selected_base_cpu = initial_env_base_cpu
		if selected_base_cpu is None:
			try:
				selected_base_cpu = gaussians.envlight.base.detach().cpu().clone()
			except Exception:
				selected_base_cpu = None
		if selected_base_cpu is not None:
			with torch.no_grad():
				try:
					gaussians.envlight.base = selected_base_cpu.to(device)
					gaussians.envlight.build_mips()
				except Exception:
					pass
				try:
					gaussians_original.envlight.base = selected_base_cpu.to(device)
					gaussians_original.envlight.build_mips()
				except Exception:
					pass

		# --- STAGE 1: Render Multi-Weather Images for Offline Evaluation ---
		final_full_img_dirs_mw = {}
		if bool(getattr(args, 'save_final_full_images_mw', True)):
			print("[消息] [最终评估] 正在渲染多天气（跨光）最终图片用于离线评估...")
			final_full_img_dirs_mw = render_and_save_final_images_mw(
				full_cameras, gaussians, pipe, bg, args, dataset,
				gaussians_original, None, save_dir, 'full'
			)
		else:
			print("[消息] [最终评估] 已跳过渲染多天气图片，将不会执行离線評估。")

		# --- (NEW) Also export evaluation_results_rpga.txt (same format as evaluate_img_rpga.py) ---
		# This evaluates final_*_images_* folders under save_dir and writes a unified summary txt.
		try:
			eval_txt_path = save_dir / 'evaluation_results_rpga.txt'
			script_path = Path('/workspace/RGA/evaluate_img_rpga.py')
			anno_dir = Path(dataset.source_path) / 'annos'
			mmdet_base = Path('/workspace/RGA/mmdet_files')

			if script_path.is_file() and anno_dir.is_dir() and mmdet_base.is_dir():
				cmd = [
					sys.executable, str(script_path),
					'--exp_dir', str(save_dir),
					'--anno_dir', str(anno_dir),
					'--mmdet_base', str(mmdet_base),
					'--device', str(args.device),
					'--all_detectors',
					'--score_thresh', str(getattr(args, 'score_thresh', 0.5)),
					'--output_file', str(eval_txt_path),
				]
				print(f"[消息] [最终评估] 正在生成汇总文件: {eval_txt_path.name}")
				subprocess.run(cmd, check=False)
				if eval_txt_path.is_file():
					print(f"[消息] [最终评估] 已生成: {eval_txt_path}")
				else:
					print("[警告] [最终评估] evaluate_img_rpga.py 未生成 evaluation_results_rpga.txt（请检查脚本输出日志）。")
			else:
				print(
					f"[警告] [最终评估] 跳过生成 evaluation_results_rpga.txt："
					f"script={script_path.is_file()}, anno_dir={anno_dir.is_dir()}, mmdet_base={mmdet_base.is_dir()}"
				)
		except Exception as e:
			print(f"[警告] [最终评估] 生成 evaluation_results_rpga.txt 失败: {e}")

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

				# Evaluate Multi-Weather Directories from saved images
				mw_results = []
				if not final_full_img_dirs_mw:
					print("  - [结果] 未渲染任何多天气图片，跳过评估。")
				else:
					for weather_name, img_dir in final_full_img_dirs_mw.items():
						asr_f_w, succ_f_w, total_f_w, ap50_f_w = evaluate_from_saved_images(
							curr_detector, img_dir, Path(dataset.source_path) / 'annos', args
						)
						mw_results.append((weather_name, 'full', asr_f_w, succ_f_w, total_f_w, ap50_f_w))
				
				# Log results
				print(f"  - [结果] 检测器: {det_name}")
				with open(log_file_path, 'a', encoding='utf-8') as f:
					f.write(f"Detector: {det_name}\n")
					# Write MW results
					if mw_results:
						f.write("  - [Multi-Weather Evaluation on Full Set]\n")
						for (wname, split, asr_w, succ_w, total_w, ap50_w) in mw_results:
							print(f"    * {wname}: ASR={asr_w:.4f}")
							f.write(f"    * {wname} [{split}] ASR: {asr_w:.4f} ({succ_w}/{total_w}), AP@0.5: {ap50_w:.4f}\n")
					else:
						f.write("  - No multi-weather images were rendered for evaluation.\n")
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
