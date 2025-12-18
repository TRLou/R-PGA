from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import time
from datetime import datetime, timezone, timedelta
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
from utils.main_utils import load_labelme_annotation, compute_adv_total_loss, coco_classes, calculate_ap_for_target_class
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector_custom
from lbm_relit import LBMRelighter
import random
import matplotlib.pyplot as plt
import copy
from utils.log_utils import TrainingLogger
from attack_options import get_attack_args


# torch.autograd.set_detect_anomaly(True)

# =================================================================================
# Reused Constants from evaluate_img.py (or could import them)
# =================================================================================
DETECTOR_PATHS = {
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
    },
    'mask-rcnn': {
        'config': 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
        'ckpt': 'checkpoints/mask_rcnn_r50_fpn_1x_coco.pth'
    },
    'yolov3': {
        'config': 'configs/yolo/yolov3_d53_8xb8-amp-ms-608-273e_coco.py',
        'ckpt': 'checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco.pth'
    },
    'pvt':{
        'config': 'configs/pvt/retinanet_pvt-m_fpn_1x_coco.py',
        'ckpt': 'checkpoints/retinanet_pvt-m_fpn_1x_coco.pth'
    },
    'detr':{
        'config': 'configs/detr/detr_r50_8xb2-150e_coco.py',
        'ckpt': 'checkpoints/detr_r50_8xb2-150e_coco.pth'
    }
}


def save_image_rgb01(img: torch.Tensor, path: Path) -> None:
	# img: (3,H,W), [0,1]
	path.parent.mkdir(parents=True, exist_ok=True)
	arr = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
	Image.fromarray(arr).save(path)


def save_visualization_grid(save_path: Path, images_dict: dict):
	"""
	Creates a grid of images and saves it.
	Args:
		save_path: Path to save the combined image.
		images_dict: A dictionary where keys are names and values are torch.Tensor (C,H,W) or PIL.Image.
	"""
	images = []
	for name, img_data in images_dict.items():
		if img_data is None:
			continue
			
		if isinstance(img_data, torch.Tensor):
			# Convert tensor to PIL Image
			if img_data.dim() == 4: # (1,C,H,W) -> (C,H,W)
				img_data = img_data.squeeze(0)
			if img_data.shape[0] == 1: # Grayscale (1,H,W) -> (H,W) -> RGB
				img_data = img_data.squeeze(0)
				pil_img = Image.fromarray((img_data.cpu().numpy() * 255).astype(np.uint8)).convert('RGB')
			else: # RGB (3,H,W)
				pil_img = Image.fromarray((img_data.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
		elif isinstance(img_data, Image.Image):
			pil_img = img_data.convert('RGB')
		else:
			warnings.warn(f"Unsupported image type for visualization: {type(img_data)} for key '{name}'")
			continue

		# Add text to image
		draw = ImageDraw.Draw(pil_img)
		try:
			# Use a common font that is likely to be available in a docker container
			font = ImageFont.truetype("DejaVuSans.ttf", 150)
		except IOError:
			try:
				font = ImageFont.truetype("dejavu-sans/DejaVuSans-Bold.ttf", 60)
			except IOError:
				font = ImageFont.load_default()
		draw.text((10, 10), name, fill="red", font=font)
		images.append(pil_img)

	if not images:
		return

	# Create grid, assuming all images are the same size
	if len(images) > 0:
		width, height = images[0].size
		cols = 3
		rows = (len(images) + cols - 1) // cols
		grid_img = Image.new('RGB', (width * cols, height * rows))
		
		for i, img in enumerate(images):
			row = i // cols
			col = i % cols
			grid_img.paste(img.resize((width, height)), (col * width, row * height))

		save_path.parent.mkdir(parents=True, exist_ok=True)
		grid_img.save(save_path)


def compute_batch_loss(cam_batch: List, gaussians: GaussianModel, pipe: dict, bg: torch.Tensor, global_step: int, args: argparse.Namespace, dataset: ModelParams, gaussians_original: GaussianModel, relighter: LBMRelighter, detector, save_dir: Path, epoch: int, batch_idx: int):
	"""
	Renders a batch of cameras, performs detection, and computes the adversarial loss.
	This function encapsulates the forward pass logic.
	"""
	imgs_for_det = []
	gt_bboxes_batch = []
	view_names_batch = []
	detect_imgs = []
	vis_data_batch = []

	for cam in cam_batch:
		name = cam.image_name
		anno_path = Path(dataset.source_path) / 'annos' / f'{name}.json'
		if not anno_path.exists():
			continue

		gt_bbox, _ = load_labelme_annotation(str(anno_path))
		
		render_pkg = render(
			cam, gaussians, pipe, bg, iteration=global_step, is_train=True, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
		)
		rgb = render_pkg['render']  # (3,H,W) rgb是带有对抗纹理的白底
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

		if red_mask_path and full_mask_path:
			red_mask_img = Image.open(red_mask_path).convert('L')
			full_mask_img = Image.open(full_mask_path).convert('L')
			red_mask = torch.from_numpy(np.array(red_mask_img)).to(device=rgb.device, dtype=torch.float32) / 255.0
			full_mask = torch.from_numpy(np.array(full_mask_img)).to(device=rgb.device, dtype=torch.float32) / 255.0
			red_mask = F.interpolate(red_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
			full_mask = F.interpolate(full_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
			other_mask = torch.clamp(1 - red_mask, 0.0, 1.0)
			red_mask3 = red_mask.repeat(3, 1, 1)
			other_mask3 = other_mask.repeat(3, 1, 1)
			full_mask3 = full_mask.repeat(3, 1, 1)
 
			with torch.no_grad():
				render_pkg_ori = render(
					cam, gaussians_original, pipe, bg, iteration=global_step, is_train=True, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
				)
				rgb_ori = render_pkg_ori['render']  # rgb_ori是不带对抗纹理的白底
			
			# 先在白色背景上，用 red_mask 混合出完整的对抗车辆
			full_adv_car = rgb_ori * (1 - red_mask3) + rgb * red_mask3

			relit_image_pil = None
			source_image = None
			source_image_path = first_existing([
				Path(dataset.source_path) / 'ori' / f'{name}.jpg',
				Path(dataset.source_path) / 'ori' / f'{name}.png'
			])
			if source_image_path:
				source_image = Image.open(source_image_path).convert('RGB')
				source_image = source_image.resize((W, H), Image.BILINEAR)
				if relighter is not None and full_mask_path:
					with torch.no_grad():
						fg_relight_image = Image.fromarray((rgb_ori.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
						bg_mask = Image.open(full_mask_path)
						relit_image_pil = relighter.relight(
							source_image=source_image, fg_relight_image=fg_relight_image, bg_mask=bg_mask,
							width=W, height=H, invert_mask=True
						)
				else:
					relit_image_pil = source_image
			
			if relit_image_pil is not None:
				# 使用 full_mask (车辆的精确轮廓) 将对抗车辆合成到重打光的背景上
				relit_image_tensor = torch.from_numpy(np.array(relit_image_pil)).permute(2,0,1).to(device=rgb.device, dtype=torch.float32) / 255.0
				detect_img_chw = relit_image_tensor * (1 - full_mask3) + full_adv_car * full_mask3
			else:
				# 如果没有重打光的背景，则使用原始渲染图 rgb_ori 作为背景
				# 同样使用 full_mask 来确保合成的边界是准确的
				detect_img_chw = rgb_ori * (1 - full_mask3) + full_adv_car * full_mask3
		else:
			detect_img_chw = rgb
		
		# Collect visualization data for this item
		vis_item = {
			'rgb': rgb.detach().clone(),
			'rgb_ori': rgb_ori.detach().clone() if 'rgb_ori' in locals() else None,
			'source_image': source_image.copy() if source_image is not None else None,
			'relit_image_pil': relit_image_pil.copy() if relit_image_pil is not None else None,
			'other_mask3': other_mask3.detach().clone() if 'other_mask3' in locals() else None,
			'detect_img_chw': detect_img_chw.detach().clone()
		}
		vis_data_batch.append(vis_item)
		
		img_for_det = detect_img_chw.permute(1, 2, 0) * 255.0
		imgs_for_det.append(img_for_det)
		gt_bboxes_batch.append(torch.from_numpy(gt_bbox).to(rgb.device))
		view_names_batch.append(name)
		detect_imgs.append(detect_img_chw)

	if not imgs_for_det:
		return None, None, None, None, None
	
	imgs_for_det_batch = torch.stack(imgs_for_det, dim=0)
	
	# =================================================================================
	# START: Temporary visualization code for detector inputs.
	# To disable, comment out or delete this block.
	# =================================================================================
	try:
		temp_vis_dir = save_dir / 'temp_imgs_for_det'
		temp_vis_dir.mkdir(parents=True, exist_ok=True)
		for i, img_tensor in enumerate(imgs_for_det):
			img_name = view_names_batch[i]
			# tensor is (H, W, 3), float, [0, 255]
			img_array = img_tensor.detach().cpu().numpy().astype(np.uint8)
			save_name = f'epoch_{epoch:03d}_batch_{batch_idx:04d}_{img_name}.png'
			Image.fromarray(img_array).save(temp_vis_dir / save_name)
	except Exception as e:
		print(f"[警告] 临时的可视化代码保存失败: {e}")
	# =================================================================================
	# END: Temporary visualization code.
	# =================================================================================
	
	# Visualize detector inputs before inference
	det_vis_dir = save_dir / f'epoch_{epoch:03d}' / 'det_vis'
	
	preds = inference_detector_custom(
		detector, imgs_for_det_batch, gt_bboxes_list=gt_bboxes_batch,
		vis_dir=None
	)

	batch_total_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	batch_cls_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	batch_reg_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	for i, pred in enumerate(preds):
		pred_instances = pred.pred_instances
		score_mask = pred_instances.scores >= 0.5
		pred_bboxes_filtered = pred_instances.bboxes[score_mask]
		pred_scores_filtered = pred_instances.scores[score_mask]
		pred_classes_filtered = pred_instances.labels[score_mask]

		gt_bbox_i = gt_bboxes_batch[i]
		if gt_bbox_i.dim() == 2 and gt_bbox_i.shape[0] > 1:
			gt_bbox_i = gt_bbox_i[:1]

		loss, cls_loss, reg_loss, _ = compute_adv_total_loss(
			pred_bboxes=pred_bboxes_filtered, pred_scores=pred_scores_filtered,
			pred_classes=pred_classes_filtered, gt_bbox=gt_bbox_i, target_class_idx=2
		)
		if not loss.requires_grad:
			# print("loss not requires grad")
			loss = loss + (imgs_for_det_batch.float().sum() * 0.0)
		batch_total_loss += loss
		batch_cls_loss += cls_loss
		batch_reg_loss += reg_loss
	
	return batch_total_loss, batch_cls_loss, batch_reg_loss, detect_imgs, vis_data_batch


@torch.no_grad()
def evaluate(test_cameras, gaussians, pipe, bg, args, dataset, gaussians_original, relighter, detector, epoch, save_dir, set_name='test'):
	"""
	Evaluates the model on the test set and calculates the Attack Success Rate (ASR).
	"""
	print(f"\n[消息] [第 {epoch}/{args.epochs} 輪] 开始在 {set_name} 集上评估...")
	
	total_attacks = 0
	successful_attacks = 0
	all_preds_for_map = []
	all_gts_for_map = []

	# Create a directory for this epoch's evaluation visualizations
	eval_vis_dir = save_dir / f"eval_{set_name}_epoch_{epoch:03d}"
	eval_vis_dir.mkdir(parents=True, exist_ok=True)
	print(f"[消息] 评估可视化结果将保存到: {eval_vis_dir}")

	cam_batches = [test_cameras[i:i + args.batch_size] for i in range(0, len(test_cameras), args.batch_size)]
	pbar_eval = tqdm(cam_batches, desc=f"Epoch {epoch} Evaluation on {set_name}", ncols=120)

	for cam_batch in pbar_eval:
		imgs_for_det = []
		view_names_with_anno = []
		gt_bboxes_batch = []
		gt_labels_batch = []

		for cam in cam_batch:
			name = cam.image_name
			anno_path = Path(dataset.source_path) / 'annos' / f'{name}.json'
			if not anno_path.exists():
				continue

			gt_bbox, gt_label_name = load_labelme_annotation(str(anno_path))
			if gt_bbox is None:
				continue
			
			try:
				gt_label_idx = coco_classes.index(gt_label_name)
			except ValueError:
				continue

			gt_bboxes_batch.append(gt_bbox)
			gt_labels_batch.append(np.array([gt_label_idx] * len(gt_bbox)))
			view_names_with_anno.append(name)

			render_pkg = render(
				cam, gaussians, pipe, bg, iteration=60000, is_train=False, 
				second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
			)
			rgb = render_pkg['render']
			H, W = rgb.shape[-2], rgb.shape[-1]
			
			red_mask_path = first_existing([
				Path(dataset.source_path) / 'red_masks' / f'{name}_mask.png',
				Path(dataset.source_path) / 'red_masks' / f'{name}_mask.jpg',
			])
			full_mask_path = first_existing([
				Path(dataset.source_path) / 'masks' / f'{name}_mask.png',
				Path(dataset.source_path) / 'masks' / f'{name}_mask.jpg',
			])

			detect_img_chw = rgb # Default
			if red_mask_path and full_mask_path:
				red_mask = torch.from_numpy(np.array(Image.open(red_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
				full_mask = torch.from_numpy(np.array(Image.open(full_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
				red_mask = F.interpolate(red_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
				full_mask = F.interpolate(full_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
				other_mask = torch.clamp(1 - red_mask, 0.0, 1.0)
				red_mask3 = red_mask.repeat(3, 1, 1)
				other_mask3 = other_mask.repeat(3, 1, 1)
				full_mask3 = full_mask.repeat(3, 1, 1)

				render_pkg_ori = render(cam, gaussians_original, pipe, bg, iteration=60000, is_train=False)
				rgb_ori = render_pkg_ori['render']
				
				# 先在白色背景上，用 red_mask 混合出完整的对抗车辆
				full_adv_car = rgb_ori * (1 - red_mask3) + rgb * red_mask3
				
				relit_image_pil = None
				source_image_path = first_existing([
					Path(dataset.source_path) / 'ori' / f'{name}.jpg',
					Path(dataset.source_path) / 'ori' / f'{name}.png'
				])
				if source_image_path and relighter:
					source_image = Image.open(source_image_path).convert('RGB').resize((W, H), Image.BILINEAR)
					fg_relight_image = Image.fromarray((rgb_ori.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
					bg_mask = Image.open(full_mask_path)
					relit_image_pil = relighter.relight(
						source_image=source_image, fg_relight_image=fg_relight_image, bg_mask=bg_mask,
						width=W, height=H, invert_mask=True
					)
				
				if relit_image_pil:
					relit_image_tensor = torch.from_numpy(np.array(relit_image_pil)).permute(2,0,1).to(device=rgb.device, dtype=torch.float32) / 255.0
					detect_img_chw = relit_image_tensor * (1 - full_mask3) + full_adv_car * full_mask3
				else:
					detect_img_chw = rgb_ori * (1 - full_mask3) + full_adv_car * full_mask3
			
			img_for_det = detect_img_chw.permute(1, 2, 0) * 255.0
			imgs_for_det.append(img_for_det)

		if not imgs_for_det:
			continue
		
		total_attacks += len(imgs_for_det)
		imgs_for_det_batch = torch.stack(imgs_for_det, dim=0)
		preds = inference_detector_custom(
			detector, imgs_for_det_batch, gt_bboxes_list=None, 
			vis_dir=eval_vis_dir, view_names=view_names_with_anno
		)

		for pred in preds:
			pred_instances = pred.pred_instances
			target_class_idx = 2
			score_mask = pred_instances.scores >= 0.5
			class_mask = pred_instances.labels == target_class_idx
			
			if not (score_mask & class_mask).any():
				successful_attacks += 1

			# Prepare predictions for mAP calculation
			num_classes = len(detector.CLASSES)
			pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
			for i in range(num_classes):
				class_indices = (pred.pred_instances.labels == i)
				if class_indices.any():
					boxes = pred.pred_instances.bboxes[class_indices].cpu().numpy()
					scores = pred.pred_instances.scores[class_indices].cpu().numpy()
					pred_for_map[i] = np.hstack([boxes, scores[:, np.newaxis]])
			all_preds_for_map.append(pred_for_map)
		
		# Collect ground truths for mAP calculation
		for bboxes, labels in zip(gt_bboxes_batch, gt_labels_batch):
			all_gts_for_map.append({'bboxes': bboxes, 'labels': labels})

		current_asr = successful_attacks / total_attacks if total_attacks > 0 else 0.0
		pbar_eval.set_postfix(ASR=f"{current_asr:.4f}")

	if total_attacks == 0:
		return 0.0, 0, 0, 0.0
		
	final_asr = successful_attacks / total_attacks
	
	# Calculate AP@0.5
	ap50 = 0.0
	if all_preds_for_map and all_gts_for_map:
		eval_results = calculate_ap_for_target_class(all_preds_for_map, all_gts_for_map, target_class_idx, iou_thr=0.5)
		ap50 = eval_results.get('AP50', 0.0)

	print(f"[消息] [第 {epoch}/{args.epochs} 輪] 在 {set_name} 集上评估完成. 平均攻击成功率 (ASR): {final_asr:.4f}, AP@0.5: {ap50:.4f}")
	return final_asr, successful_attacks, total_attacks, ap50


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
		
		# Define full dataset
		full_cameras = train_cameras + test_cameras
		
		# Log file init
		log_file_path = save_dir / 'training_log.txt'
		with open(log_file_path, 'a', encoding='utf-8') as f:
			f.write("\n\n==================================================\n")
			f.write(f"=== Final Cross-Detector Evaluation Results (Epoch {args.epochs}) ===\n")
			f.write("==================================================\n")
			f.write(f"Timestamp: {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

		# Release the training detector
		del detector
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			
		# Iterate over all detectors
		base_path = Path('/workspace/RGA/mmdet_files')
		
		for det_name, det_info in DETECTOR_PATHS.items():
			print(f"\n>>> [评估] 正在切换到检测器: {det_name} ...")
			
			try:
				# Init detector
				cfg_path = str(base_path / det_info['config'])
				ckpt_path = str(base_path / det_info['ckpt'])
				
				curr_detector = init_detector(cfg_path, ckpt_path, device=device)
				if not hasattr(curr_detector, 'CLASSES'):
					curr_detector.CLASSES = coco_classes
				
				# 1. Evaluate on Test Set (Group 1)
				print(f"    -> 正在测试集上评估 ({len(test_cameras)} 张)...")
				asr_test, succ_test, total_test, ap50_test = evaluate(
					test_cameras, gaussians, pipe, bg, args, dataset, 
					gaussians_original, relighter, curr_detector, args.epochs, save_dir, f'final_test_{det_name}'
				)
				
				# 2. Evaluate on Full Set (Group 2)
				print(f"    -> 正在全集(训练+测试)上评估 ({len(full_cameras)} 张)...")
				# Note: To save time, we reuse the numbers from test set evaluation if possible, 
				# but here we just run evaluate() again on full set for simplicity and correctness 
				# (in case train set wasn't evaluated fully before).
				asr_full, succ_full, total_full, ap50_full = evaluate(
					full_cameras, gaussians, pipe, bg, args, dataset, 
					gaussians_original, relighter, curr_detector, args.epochs, save_dir, f'final_full_{det_name}'
				)
				
				# Log results
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
