from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import time
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
import envlight
import torchvision
from envlight.utils import cubemap_to_latlong
from detectors.mmdet_wrapper import MMDetLoss, CocoGtLookup
from utils.main_utils import load_labelme_annotation, compute_adv_total_loss, coco_classes
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector_custom
from lbm_relit import LBMRelighter


# torch.autograd.set_detect_anomaly(True)


def save_image_rgb01(img: torch.Tensor, path: Path) -> None:
	# img: (3,H,W), [0,1]
	path.parent.mkdir(parents=True, exist_ok=True)
	arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
	Image.fromarray(arr).save(path)


def load_hdr_image_and_tonemap(hdr_path, gamma=2.2):
    """Loads and tonemaps an HDR image using OpenCV, returning a PIL Image."""
    try:
        # 1. Load HDR image using OpenCV
        hdr_cv = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr_cv is None:
            print(f"无法加载HDR文件: {hdr_path}")
            return None
        
        # 2. Convert from BGR (OpenCV default) to RGB
        hdr_cv = cv2.cvtColor(hdr_cv, cv2.COLOR_BGR2RGB)

        # Ensure data is float32 for calculations, as in the reference script
        if hdr_cv.dtype != np.float32:
            hdr_cv = hdr_cv.astype(np.float32)

        # 3. Manual tonemapping based on the logic from batch_visualize_hdri.py
        # This approach often handles highlights more gracefully than the default cv2.TonemapReinhard
        luminance = 0.2126 * hdr_cv[..., 0] + 0.7152 * hdr_cv[..., 1] + 0.0722 * hdr_cv[..., 2]
        # Add a small epsilon to prevent division by zero, although luminance is unlikely to be -1
        scale = 1.0 / (1.0 + luminance[..., np.newaxis] + 1e-6)
        ldr_cv = hdr_cv * scale

        # 4. Apply gamma correction
        ldr_cv = np.power(ldr_cv, 1.0/gamma)

        # 5. Convert the LDR numpy array [0, 1] to a PIL Image
        ldr_8bit = np.clip(ldr_cv * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(ldr_8bit)

    except Exception as e:
        print(f"处理HDR文件 '{hdr_path}' 时出错: {e}")
        return None


def latest_iteration_ply(scene_path: Path) -> Path:
	pc_dir = scene_path / 'point_cloud'
	assert pc_dir.is_dir(), f"point_cloud not found under {scene_path}"
	iters = []
	for p in pc_dir.iterdir():
		if p.is_dir() and p.name.startswith('iteration_'):
			try:
				iters.append((int(p.name.split('_')[-1]), p))
			except Exception:
				pass
	assert iters, f"no iteration_* folders in {pc_dir}"
	iters.sort(key=lambda x: x[0])
	return iters[-1][1] / 'point_cloud.ply'


def latest_checkpoint_pth(scene_path: Path) -> Path | None:
	"""Return Path to latest chkpnt<iter>.pth file in a directory, or None if not found."""
	assert scene_path.is_dir(), f"Scene path not found: {scene_path}"
	iters = []
	for p in scene_path.glob('chkpnt*.pth'):
		try:
			num = int(p.stem.replace('chkpnt', ''))
			iters.append((num, p))
		except Exception:
			pass
	if not iters:
		return None
	iters.sort(key=lambda x: x[0])
	return iters[-1][1]


def iteration_ply(scene_path: Path, iteration: int) -> Path:
	"""Return PLY path for a specific iteration, or latest if iteration < 0 (GIR style)."""
	if iteration is None or int(iteration) < 0:
		return latest_iteration_ply(scene_path)
	pc_dir = scene_path / 'point_cloud' / f'iteration_{int(iteration)}'
	ply_path = pc_dir / 'point_cloud.ply'
	assert ply_path.is_file(), f"PLY not found: {ply_path}"
	return ply_path


def first_existing(paths):
	for p in paths:
		if p and Path(p).is_file():
			return str(p)
	return None


def main():
	# =================================================================================
	# 1. 参数解析
	# =================================================================================
	parser = argparse.ArgumentParser("RGA physical attack baseline (GIR + MMDet)")
	model_params = ModelParams(parser, sentinel=False)
	pipeline_params = PipelineParams(parser)

	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--detector', type=str, default='yolox', choices=['yolov5', 'yolox', 'faster-rcnn', 'mask-rcnn', 'd-detr'])
	parser.add_argument('--anno_dir', type=str, default=None, help='Directory for annotation files (e.g., annos)')
	parser.add_argument('--iteration', type=int, default=-1, help='Iteration to load; -1 means latest (GIR style)')
	parser.add_argument('--hdr_rotation', action='store_true', default=False)
	parser.add_argument('--second_stage_step', type=int, default=30000)
	parser.add_argument('--environment_texture', type=str, default="")
	parser.add_argument('--environment_scale', type=float, default=1.0)
	parser.add_argument('--max_cams', type=int, default=0, help='Limit number of cameras for quick test (0=all)')
	parser.add_argument('--lr', type=float, default=1)
	parser.add_argument('--save_dir', type=str, default='RGA_output')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size for attack.')
	# LBM relighting arguments
	parser.add_argument('--enable_lbm_relight', default=True, action='store_true', help='Enable LBM background relighting.')
	parser.add_argument('--lbm_ckpt_dir', type=str, default='/workspace/lbm/checkpoints', help='Path to LBM checkpoints directory.')
	parser.add_argument('--enable_random_hdr', default=True, action='store_true', help='Enable dynamic environment relighting during attack.')
	parser.add_argument('--hdri_dir', type=str, default='hdri', help='Directory containing additional HDR environment light models (.pth files).')

	args = get_combined_args(parser)

	# =================================================================================
	# 2. 环境与路径设置
	# =================================================================================
	device = torch.device(args.device)
	save_dir = Path(args.save_dir)
	# Resolve relative save_dir to repository directory to avoid cwd permission issues
	if not save_dir.is_absolute():
		repo_dir = Path(__file__).resolve().parent
		save_dir = repo_dir / save_dir
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

	# =================================================================================
	# 3. 加载场景与高斯模型
	# =================================================================================
	dataset = model_params.extract(args)
	if args.environment_texture != "":
		gaussians = GaussianModel(dataset.sh_degree, environment_texture=args.environment_texture, environment_scale=args.environment_scale)
	else:
		gaussians = GaussianModel(dataset.sh_degree)

	env_lights_list = [] # List to store (name, envlight_object, file_path) tuples

	if args.environment_texture == "":
		print("[消息] 未提供 environment_texture，将尝试从最新的检查点加载 'envlight'...")
		model_dir = Path(dataset.model_path)
		latest_ckpt_path = latest_checkpoint_pth(model_dir)
		if latest_ckpt_path:
			print(f"[消息] 找到最新检查点: {latest_ckpt_path}")
			try:
				ckpt_data_tuple, _ = torch.load(str(latest_ckpt_path), map_location=device)
				envlight_state_dict = None
				for item in ckpt_data_tuple:
					if isinstance(item, dict):
						if 'state' in item and 'param_groups' in item:
							continue
						else:
							envlight_state_dict = item
							break # Found it
				
				if envlight_state_dict is not None:
					print("[消息] 正在从检查点加载 'env_light'...")
					# Load the state into the default envlight object that comes with the gaussians model
					gaussians.envlight.load_state_dict(envlight_state_dict)
					# Add the now-configured envlight object to our list of choices
					env_lights_list.append(('original_from_ckpt', gaussians.envlight, None))
					print("[消息] 成功加载 'envlight' 并将其设为默认/备选光照。")

					# --- Start: Verification Save ---
					try:
						print("[消息] 正在保存加载的 envlight 到图片以供验证...")
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
	else:
		# If a texture is provided via args, it's already in our main `gaussians` object.
		# Add this envlight object to our list of choices.
		print("[消息] 使用 --environment_texture 提供的环境光作为默认/备选光照。")
		env_lights_list.append((Path(args.environment_texture).name, gaussians.envlight, args.environment_texture))


	if args.enable_random_hdr:
		print(f"[消息] 已启用动态重打光。正在从 '{args.hdri_dir}' 加载额外的环境贴图...")
		hdri_dir = Path(args.hdri_dir)
		if hdri_dir.is_dir():
			extensions = ['*.hdr', '*.exr', '*.png', '*.jpg', '*.jpeg']
			hdri_paths = []
			for ext in extensions:
				hdri_paths.extend(hdri_dir.glob(ext))

			print(f"[消息] 找到 {len(hdri_paths)} 个环境贴图文件。")
			for hdri_path in hdri_paths:
				try:
					print(f"[消息] 正在从贴图文件创建环境光: {hdri_path.name}")
					# Directly create a new EnvLight object for each HDR file
					# The texture path is the first positional argument, not a keyword argument.
					new_envlight = envlight.EnvLight(
						str(hdri_path), 
						scale=1.0, 
						min_res=16, max_res=512, 
						min_roughness=0.08, max_roughness=0.5, 
						trainable=True
					).cuda()
					env_lights_list.append((hdri_path.name, new_envlight, str(hdri_path)))
					print(f"[消息] 成功创建并添加环境光: {hdri_path.name}")
				except Exception as e:
					print(f"[警告] 从 '{hdri_path.name}' 创建环境光失败: {e}")
		else:
			print(f"[警告] HDRI 目录 '{args.hdri_dir}' 未找到或不是一个目录。")
	
	if len(env_lights_list) > 1:
		print(f"[消息] 总共加载了 {len(env_lights_list)} 个环境光照条件，将在训练中随机使用。")
	elif not env_lights_list:
		print("[警告] 未加载任何环境光。模型将使用其默认的内部环境光（如果有）。")

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
	optimizer = torch.optim.Adam([gaussians._albedo_init], lr=args.lr)  # 暂停 albedo 优化，后续需要时可恢复
	
	# gaussians.envlight.train()
	# for p in gaussians.envlight.net.parameters():
	# 	p.requires_grad = False
	# optimizer = torch.optim.Adam(
	# 	[
	# 		{'params': [gaussians.envlight.init_base], 'lr': args.lr},
	# 		{'params': [gaussians.envlight.base_train], 'lr': args.lr},
	# 	]
	# )

	# =================================================================================
	# 5. 初始化目标检测器与损失函数
	# =================================================================================
	print("[消息] 正在初始化检测器...")

	base_path = Path('/workspace/RGA/mmdet_files')
	
	detector_paths = {
		'yolox': {
			'config': 'configs/yolox/yolox_l_8xb8-300e_coco.py',
			'ckpt': 'checkpoints/yolox_l_8x8_300e_coco.pth'
		},
		'faster-rcnn': {
			'config': 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
			'ckpt': 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
		},
		'd-detr': {
			'config': 'configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py',
			'ckpt': 'checkpoints/d-detr.pth'
		}
	}

	selected_detector = detector_paths.get(args.detector)

	if selected_detector is None:
		raise ValueError(f"Detector '{args.detector}' not found in hardcoded paths. Available: {list(detector_paths.keys())}")

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
	for epoch in range(1, args.epochs + 1):
		print(f"[消息] [第 {epoch}/{args.epochs} 轮] 开始")
		train_cameras = scene.getTrainCameras()

		cams_to_process = train_cameras
		if args.max_cams > 0:
			cams_to_process = cams_to_process[:args.max_cams]
		
		# Create batches from camera list
		cam_batches = [cams_to_process[i:i + batch_size] for i in range(0, len(cams_to_process), batch_size)]
		
		pbar = tqdm(cam_batches, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)

		for cam_batch in pbar:
			# --- Dynamic Relighting ---
			hdri_path_for_batch = None
			light_name = 'unknown' # Default value
			if env_lights_list:
				# If random is enabled and we have choices, pick one.
				if args.enable_random_hdr and len(env_lights_list) > 1:
					import random
					light_name, selected_envlight_obj, hdri_path_for_batch = random.choice(env_lights_list)
					gaussians.envlight = selected_envlight_obj # Direct replacement
					print(f"[调试] 当前envlight使用的是 {light_name} (随机)")
				# Otherwise, just use the first one (which is the original)
				else:
					light_name, selected_envlight_obj, hdri_path_for_batch = env_lights_list[0]
					gaussians.envlight = selected_envlight_obj # Direct replacement
					print(f"[调试] 当前envlight使用的是 {light_name} (固定)")
				
			# --- Batch Processing ---
			imgs_for_det = []
			gt_bboxes_batch = []
			view_names_batch = []
			detect_imgs = []

			# 1. Render all images in the batch and collect data
			start_t = time.time()
			for cam in cam_batch:
				name = cam.image_name
				anno_path = Path(dataset.source_path) / 'annos' / f'{name}.json'
				if not anno_path.exists():
					continue

				gt_bbox, _ = load_labelme_annotation(str(anno_path))
				
				render_pkg = render(
					cam, gaussians, pipe, bg, iteration=global_step, is_train=True, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
				)
				rgb = render_pkg['render']  # (3,H,W)
				# 仅 red_mask 区域允许梯度回传，其余区域使用脱梯度的拷贝
				H, W = rgb.shape[-2], rgb.shape[-1]
				red_mask_path = first_existing([
					Path(dataset.source_path) / 'red_masks' / f'{name}_mask.png',
					Path(dataset.source_path) / 'red_masks' / f'{name}_mask.jpg',
					Path(dataset.source_path) / 'red_masks' / f'{name}_mask.jpeg',
					Path(dataset.source_path) / 'red_masks' / f'{name}_mask.bmp',
				])
				full_mask_path = first_existing([
					Path(dataset.source_path) / 'masks' / f'{name}_mask.png',
					Path(dataset.source_path) / 'masks' / f'{name}_mask.jpg',
					Path(dataset.source_path) / 'masks' / f'{name}_mask.jpeg',
					Path(dataset.source_path) / 'masks' / f'{name}_mask.bmp',
				])
				# print(f"[消息] red_mask_path: {red_mask_path}")
				# print(f"[消息] full_mask_path: {full_mask_path}")
				# print(f"[消息] dataset.source_path: {Path(dataset.source_path)/ 'red_masks'/ f'{name}.png'} ")
				# print(Path(dataset.source_path) / 'red_masks' / f'{name}.png')
				if red_mask_path and full_mask_path:
					# 读取并缩放到渲染分辨率，范围[0,1]，形状(1,H,W)
					red_mask_img = Image.open(red_mask_path).convert('L')
					full_mask_img = Image.open(full_mask_path).convert('L')
					red_mask = torch.from_numpy(np.array(red_mask_img)).to(device=device, dtype=torch.float32) / 255.0
					full_mask = torch.from_numpy(np.array(full_mask_img)).to(device=device, dtype=torch.float32) / 255.0
					red_mask = red_mask.unsqueeze(0).unsqueeze(0)  # 1x1xH0xW0
					full_mask = full_mask.unsqueeze(0).unsqueeze(0)
					red_mask = F.interpolate(red_mask, size=(H, W), mode='nearest')
					full_mask = F.interpolate(full_mask, size=(H, W), mode='nearest')
					red_mask = red_mask.squeeze(0)   # 1xHxW
					full_mask = full_mask.squeeze(0) # 1xHxW
					other_mask = torch.clamp(1 - red_mask, 0.0, 1.0)  # 暂时是 1-red_mask, lbm加入后是full_mask - red_mask
					red_mask3 = red_mask.repeat(3, 1, 1)
					other_mask3 = other_mask.repeat(3, 1, 1)

					# 由于需要relit输入原始车辆，因此用未修改过的gaussians渲染一次。
					with torch.no_grad():
						render_pkg_ori = render(
							cam, gaussians_original, pipe, bg, iteration=global_step, is_train=True, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
						)
						rgb_ori = render_pkg_ori['render']

					# --- LBM Relighting Logic ---
					relit_image_pil = None
					source_image = None
					
					source_image_path = first_existing([
						Path(dataset.source_path) / 'ori' / f'{name}.jpg',
						Path(dataset.source_path) / 'ori' / f'{name}.png'
					])

					if source_image_path:
						source_image = Image.open(source_image_path).convert('RGB')
						source_image = source_image.resize((W, H), Image.BILINEAR)

						# If original light is used, skip LBM and use the original image as the background.
						if light_name == 'original_from_ckpt':
							print("[消息] 使用原始光照，跳过LBM重打光，直接使用原图背景。")
							relit_image_pil = source_image
						# Otherwise, if LBM is available, use it to relight the background.
						elif relighter is not None and full_mask_path:
							with torch.no_grad():
								fg_relight_image = Image.fromarray((rgb_ori.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
								bg_mask = Image.open(full_mask_path)
								
								print("[消息] 正在调用 LBMRelighter 进行背景重打光...")
								relit_image_pil = relighter.relight(
									source_image=source_image,
									fg_relight_image=fg_relight_image,
									bg_mask=bg_mask,
									width=W,
									height=H,
									invert_mask=True # As requested
								)
					
					# After attempting to get the relit/original background, compose the final image.
					if relit_image_pil is not None:
						if source_image: # Only create viz if we have the ori image
							# 1. Save visualization of LBM relighting output with comparison
							lbm_save_dir = save_dir / f'epoch_{epoch:03d}' / 'lbm_output'
							lbm_save_dir.mkdir(parents=True, exist_ok=True)
							
							# Prepare images for composition
							lbm_output_img = relit_image_pil
							ori_img = source_image # Already loaded and resized

							# Load or create HDRI visualization
							if hdri_path_for_batch:
								# Use the new loader for .hdr files, standard PIL for others.
								if Path(hdri_path_for_batch).suffix.lower() == '.hdr':
									hdri_img = load_hdr_image_and_tonemap(hdri_path_for_batch)
								else:
									try:
										hdri_img = Image.open(hdri_path_for_batch).convert('RGB')
									except Exception as e:
										print(f"加载标准图像失败 '{hdri_path_for_batch}': {e}")
										hdri_img = None
								
								# If loading failed for any reason, create a fallback image
								if hdri_img is None:
									hdri_img = Image.new('RGB', (ori_img.width // 2, ori_img.height), (255, 0, 255)) # Magenta error indicator
							else:
								# No HDRI path was provided (e.g., using original envlight), create black placeholder
								hdri_img = Image.new('RGB', (ori_img.width // 2, ori_img.height), (0, 0, 0))

							# Resize HDRI to match height of others, maintaining aspect ratio
							target_height = ori_img.height
							w, h = hdri_img.size
							new_w = int(w * (target_height / h))
							hdri_img_resized = hdri_img.resize((new_w, target_height), Image.Resampling.LANCZOS)
							
							# Create composite image
							total_width = lbm_output_img.width + ori_img.width + hdri_img_resized.width
							composite_img = Image.new('RGB', (total_width, target_height))
							
							# Paste images side-by-side
							composite_img.paste(lbm_output_img, (0, 0))
							composite_img.paste(ori_img, (lbm_output_img.width, 0))
							composite_img.paste(hdri_img_resized, (lbm_output_img.width + ori_img.width, 0))
							
							# Add HDRI name to the visualization
							try:
								draw = ImageDraw.Draw(composite_img)
								# Use a default font that's usually available.
								font = ImageFont.load_default(size=24)
								
								text_to_draw = light_name
								
								# Position for the text (on the third image, top-left)
								text_x = lbm_output_img.width + ori_img.width + 10
								text_y = 10
								
								# Draw a semi-transparent background for better readability
								bbox = draw.textbbox((text_x, text_y), text_to_draw, font=font)
								# Add padding
								bg_bbox = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)
								draw.rectangle(bg_bbox, fill=(0, 0, 0, 128)) # Black with 50% opacity if RGBA
								
								# Draw the text
								draw.text((text_x, text_y), text_to_draw, font=font, fill=(255, 255, 255)) # White text
							except Exception as e:
								print(f"[警告] 无法在可视化图片上绘制文字: {e}")

							# Save the final composite image
							composite_img.save(str(lbm_save_dir / f'{name}.png'))

						# Perform the composition outside of no_grad to keep the gradient flow from `rgb`
						relit_image_tensor = torch.from_numpy(np.array(relit_image_pil)).permute(2,0,1).to(device, dtype=torch.float32) / 255.0
						detect_img_chw = relit_image_tensor * other_mask3 + rgb * red_mask3
					else:
						# Fallback case if relit_image_pil could not be created
						print(f"[警告] LBM/原始背景: 处理失败或源文件未找到，使用原始渲染背景。")
						detect_img_chw = rgb_ori * other_mask3 + rgb * red_mask3
					# --- End LBM Relighting Logic ---
				else:
					# 掩码缺失则直接使用原渲染图，保持可训练
					detect_img_chw = rgb
				
				# Prepare for detector: list of (H,W,3) tensors
				img_for_det = detect_img_chw.permute(1, 2, 0) * 255.0
				
				imgs_for_det.append(img_for_det)
				gt_bboxes_batch.append(gt_bbox.to(device))
				view_names_batch.append(name)
				detect_imgs.append(detect_img_chw)

			if not imgs_for_det:
				continue

			render_time = time.time() - start_t
		
			# 2. Run detector on the whole batch
			# Stack list of tensors into a single batch tensor
			imgs_for_det_batch = torch.stack(imgs_for_det, dim=0)
			# print(f'[调试] 送入检测器的堆叠张量形状: {imgs_for_det_batch.shape}')

			# Visualize detector inputs before inference
			debug_inputs_dir = save_dir / f'epoch_{epoch:03d}' / 'det_inputs'
			for i, name in enumerate(view_names_batch):
				img_chw_01 = (imgs_for_det_batch[i].permute(2, 0, 1).float() / 255.0)
				save_image_rgb01(img_chw_01, debug_inputs_dir / f'{name}.png')

			#################### 检测器 及 loss计算 ####################
			# print(f'[调试] 已保存 {imgs_for_det_batch.shape[0]} 张检测输入图像到: {debug_inputs_dir}')
			# print(f'[调试] 送入检测器的堆叠张量形状: {imgs_for_det_batch.shape}')
			preds = inference_detector_custom(
					detector,
					imgs_for_det_batch,
					gt_bboxes_list=gt_bboxes_batch,
					vis_dir=str(save_dir / f'epoch_{epoch:03d}' / 'det_vis')
				)

			# 3. Compute total loss for the batch
			batch_total_loss = torch.tensor(0.0, device=device)
			print(f"\n--- 批次损失计算开始（批大小: {len(preds)}）---")
			for i, pred in enumerate(preds):
				print(f"\n[调试] 处理批次中的第 {i} 项（视角: {view_names_batch[i]}）")
				pred_instances = pred.pred_instances
				score_threshold = 0.5
				score_mask = pred_instances.scores >= score_threshold
				
				pred_bboxes_filtered = pred_instances.bboxes[score_mask]
				pred_scores_filtered = pred_instances.scores[score_mask]
				pred_classes_filtered = pred_instances.labels[score_mask]

				# print(f"[调试] 分数阈值（{score_threshold}）后预测数量: {len(pred_bboxes_filtered)}")
				# print(f"[调试] pred_bboxes:\n{pred_bboxes_filtered.detach().cpu()}")
				# print(f"[调试] pred_scores:\n{pred_scores_filtered.detach().cpu()}")
				# print(f"[调试] pred_classes:\n{pred_classes_filtered.detach().cpu()}")
				# print(f"[调试] gt_bbox:\n{gt_bboxes_batch[i].detach().cpu()}")

				# 若存在多个GT框，取第一个（或按需策略挑选）；确保传入形状为 [1,4]
				gt_bbox_i = gt_bboxes_batch[i]
				if gt_bbox_i.dim() == 2 and gt_bbox_i.shape[0] > 1:
					print(f"[调试] 发现 {gt_bbox_i.shape[0]} 个GT框，选用第一个参与IoU计算")
					gt_bbox_i = gt_bbox_i[:1]
				# 直接使用像素坐标（与检测器输出一致），不进行归一化
				loss, _ = compute_adv_total_loss(
					pred_bboxes=pred_bboxes_filtered,
					pred_scores=pred_scores_filtered,
					pred_classes=pred_classes_filtered,
					gt_bbox=gt_bbox_i,
					target_class_idx=2
				)
				print(f"[调试] 第 {i} 项的损失: {float(loss.detach().item())}")
				# 若loss不含梯度链，附加一个对图像的0系数项，确保反传不报错（梯度仍为0）
				if not loss.requires_grad:
					loss = loss + (imgs_for_det_batch.float().sum() * 0.0)
				# 累加loss（保持标量）
				batch_total_loss = batch_total_loss + loss
			
			print(f"--- 批次损失计算结束 ---")
			#################### 检测器 及 loss计算 【结束】####################
			
			optimizer.zero_grad()
			batch_total_loss.backward()
			optimizer.step()

		# 5. Logging and saving
		avg_loss = batch_total_loss.item() / len(imgs_for_det) if len(imgs_for_det) > 0 else 0
		pbar.set_postfix(batch_loss=f"{avg_loss:.4f}", render_time=f"{render_time:.3f}s")

		# 保存 detect_img（合成后的检测输入图像）
		detect_save_dir = save_dir / f'epoch_{epoch:03d}' / 'detect_img'
		for i, name in enumerate(view_names_batch):
			img_out = detect_save_dir / f'{name}.png'
			save_image_rgb01(detect_imgs[i], img_out)
		
		global_step += len(view_names_batch)

		# save ply per-epoch
		if gaussians is not None:
			gaussians.save_ply(str(save_dir / f'point_cloud_epoch_{epoch:03d}.ply'))
			print(f"[消息] [第 {epoch}/{args.epochs} 轮] 已保存点云: point_cloud_epoch_{epoch:03d}.ply")

	# =================================================================================
	# 7. 保存最终模型
	# =================================================================================
	if gaussians is not None:
		gaussians.save_ply(str(save_dir / 'point_cloud_final.ply'))


if __name__ == '__main__':
	main()


