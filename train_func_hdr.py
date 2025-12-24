from __future__ import annotations

import warnings
import argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

# Project-specific imports
from arguments import ModelParams
from scene import GaussianModel
from gaussian_renderer import render
from utils.main_utils import load_labelme_annotation, compute_adv_total_loss, coco_classes, calculate_ap_for_target_class, compute_iou
from mmdet.apis import init_detector, inference_detector_custom, inference_detector
from lbm_relit import LBMRelighter
from mmdet.visualization import DetLocalVisualizer


# =================================================================================
# Constants
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
			cam, gaussians, pipe, bg, iteration=global_step, is_train=False, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
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
					cam, gaussians_original, pipe, bg, iteration=global_step, is_train=False, second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
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
						source_image_tensor = torch.from_numpy(np.array(source_image)).permute(2, 0, 1).to(device=rgb.device, dtype=torch.float32) / 255.0
						composite_image_tensor = source_image_tensor * (1 - full_mask3) + rgb_ori * full_mask3
						fg_relight_image = Image.fromarray((composite_image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

						bg_mask = Image.open(full_mask_path)
						
						# Extract SH coeffs from gaussians object if they exist for HDR relighting
						hdr_sh_coeffs = getattr(gaussians, 'hdr_sh_coeffs', None)

						relit_image_pil = relighter.relight_hdr(
							source_image=source_image, fg_relight_image=fg_relight_image, bg_mask=bg_mask,
							width=W, height=H, invert_mask=True, hdr_sh_coeffs=hdr_sh_coeffs
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
		return None, None, None, None, None, None
	
	imgs_for_det_batch = torch.stack(imgs_for_det, dim=0)
	
	if bool(getattr(args, 'save_temp_imgs_for_det', False)):
		try:
			temp_vis_dir = save_dir / 'temp_imgs_for_det'
			temp_vis_dir.mkdir(parents=True, exist_ok=True)
			for i, img_tensor in enumerate(imgs_for_det):
				img_name = view_names_batch[i]
				img_array = img_tensor.detach().cpu().numpy().astype(np.uint8)
				save_name = f'epoch_{epoch:03d}_batch_{batch_idx:04d}_{img_name}.png'
				Image.fromarray(img_array).save(temp_vis_dir / save_name)
		except Exception as e:
			print(f"[警告] 保存 temp_imgs_for_det 失败: {e}")
	
	preds = inference_detector_custom(
		detector, imgs_for_det_batch, gt_bboxes_list=gt_bboxes_batch,
		vis_dir=None
	)

	batch_total_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	batch_cls_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	batch_reg_loss = torch.tensor(0.0, device=imgs_for_det_batch.device)
	per_sample_total_losses: list[torch.Tensor] = []
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
			pred_classes=pred_classes_filtered, gt_bbox=gt_bbox_i, target_class_idx=2,
			reg_loss_weight=args.reg_loss_weight
		)
		if not loss.requires_grad:
			loss = loss + (imgs_for_det_batch.float().sum() * 0.0)
		per_sample_total_losses.append(loss)
		batch_total_loss += loss
		batch_cls_loss += cls_loss
		batch_reg_loss += reg_loss
	
	per_sample_total_losses_t = torch.stack(per_sample_total_losses, dim=0) if per_sample_total_losses else None
	return batch_total_loss, batch_cls_loss, batch_reg_loss, detect_imgs, vis_data_batch, per_sample_total_losses_t

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

	# Eval visualization: only save periodically to reduce I/O (default every 5 epochs).
	eval_vis_dir = None
	if str(set_name).lower() == 'test':
		interval = int(getattr(args, 'eval_vis_interval', 5))
		# Always save on the first epoch; otherwise follow interval
		if int(epoch) == 1 or (interval > 0 and (int(epoch) % interval == 0)):
			eval_vis_dir = save_dir / f"eval_{set_name}_epoch_{epoch:03d}"
			eval_vis_dir.mkdir(parents=True, exist_ok=True)
			print(f"[消息] 评估可视化结果将保存到: {eval_vis_dir} (interval={interval})")
		else:
			print(f"[消息] 本轮跳过评估可视化保存 (interval={interval})")

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
				
				full_adv_car = rgb_ori * (1 - red_mask3) + rgb * red_mask3
				
				relit_image_pil = None
				source_image_path = first_existing([
					Path(dataset.source_path) / 'ori' / f'{name}.jpg',
					Path(dataset.source_path) / 'ori' / f'{name}.png'
				])
				if source_image_path:
					source_image = Image.open(source_image_path).convert('RGB').resize((W, H), Image.BILINEAR)
					if relighter:
						fg_relight_image = Image.fromarray((rgb_ori.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
						bg_mask = Image.open(full_mask_path)

						# Extract SH coeffs from gaussians object if they exist for HDR relighting
						hdr_sh_coeffs = getattr(gaussians, 'hdr_sh_coeffs', None)

						relit_image_pil = relighter.relight_hdr(
							source_image=source_image, fg_relight_image=fg_relight_image, bg_mask=bg_mask,
							width=W, height=H, invert_mask=True, hdr_sh_coeffs=hdr_sh_coeffs
						)
					else:
						relit_image_pil = source_image
				
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

		for i, pred in enumerate(preds):
			pred_instances = pred.pred_instances
			target_class_idx = 2
			score_thresh = getattr(args, 'score_thresh', 0.5)
			score_mask = pred_instances.scores >= score_thresh
			class_mask = pred_instances.labels == target_class_idx
			
			# --- New ASR criterion: require IoU with GT >= 0.5 for failure ---
			is_attack_successful = True
			match_indices = (score_mask & class_mask).nonzero(as_tuple=False).squeeze(1)
			if match_indices.numel() > 0:
				pred_bboxes_tc = pred_instances.bboxes[match_indices]  # [K,4]
				
				# Collect GT bboxes for the target class (if labels are available)
				gt_bboxes_np = gt_bboxes_batch[i] if i < len(gt_bboxes_batch) else None
				gt_labels_np = gt_labels_batch[i] if i < len(gt_labels_batch) else None
				
				if gt_bboxes_np is not None:
					if gt_labels_np is not None:
						try:
							# Filter GTs to the target class if label info exists
							sel = (gt_labels_np == target_class_idx)
							gt_sel_np = gt_bboxes_np[sel] if sel.shape[0] == gt_bboxes_np.shape[0] else gt_bboxes_np
						except Exception:
							gt_sel_np = gt_bboxes_np
					else:
						gt_sel_np = gt_bboxes_np
					
					if gt_sel_np is not None and len(gt_sel_np) > 0:
						gt_sel_t = torch.from_numpy(gt_sel_np).to(pred_bboxes_tc.device, dtype=pred_bboxes_tc.dtype)
						# Compute max IoU across all GTs for each pred box
						max_iou = torch.zeros(pred_bboxes_tc.shape[0], device=pred_bboxes_tc.device)
						for g in gt_sel_t:
							ious = compute_iou(pred_bboxes_tc, g.unsqueeze(0))  # [K]
							max_iou = torch.maximum(max_iou, ious)
						# If any pred overlaps sufficiently, attack fails for this image
						if (max_iou >= 0.5).any():
							is_attack_successful = False
				# If no GT for target class, keep as success
			
			if is_attack_successful:
				successful_attacks += 1

			num_classes = len(detector.CLASSES)
			pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
			for i in range(num_classes):
				class_indices = (pred.pred_instances.labels == i)
				if class_indices.any():
					boxes = pred.pred_instances.bboxes[class_indices].cpu().numpy()
					scores = pred.pred_instances.scores[class_indices].cpu().numpy()
					pred_for_map[i] = np.hstack([boxes, scores[:, np.newaxis]])
			all_preds_for_map.append(pred_for_map)
		
		for bboxes, labels in zip(gt_bboxes_batch, gt_labels_batch):
			all_gts_for_map.append({'bboxes': bboxes, 'labels': labels})

		current_asr = successful_attacks / total_attacks if total_attacks > 0 else 0.0
		pbar_eval.set_postfix(ASR=f"{current_asr:.4f}")

	if total_attacks == 0:
		return 0.0, 0, 0, 0.0
		
	final_asr = successful_attacks / total_attacks
	
	ap50 = 0.0
	if all_preds_for_map and all_gts_for_map:
		eval_results = calculate_ap_for_target_class(all_preds_for_map, all_gts_for_map, target_class_idx, iou_thr=0.5)
		ap50 = eval_results.get('AP50', 0.0)

	print(f"[消息] [第 {epoch}/{args.epochs} 輪] 在 {set_name} 集上评估完成. 平均攻击成功率 (ASR): {final_asr:.4f}, AP@0.5: {ap50:.4f}")
	return final_asr, successful_attacks, total_attacks, ap50


@torch.no_grad()
def render_and_save_final_images(cameras, gaussians, pipe, bg, args, dataset, gaussians_original, relighter, save_dir: Path, set_name: str):
    """Renders adversarial images from a set of cameras and saves them to disk."""
    output_dir = save_dir / f"final_{set_name}_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[消息] 开始渲染并保存 '{set_name}' 集的图像到: {output_dir}")

    for cam in tqdm(cameras, desc=f"渲染 {set_name} 集图像", ncols=120):
        name = cam.image_name
        anno_path = Path(dataset.source_path) / 'annos' / f'{name}.json'
        if not anno_path.exists():
            continue

        render_pkg = render(
            cam, gaussians, pipe, bg, iteration=60000, is_train=False, 
            second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
        )
        rgb = render_pkg['render']
        H, W = rgb.shape[-2], rgb.shape[-1]
        
        red_mask_path = first_existing([Path(dataset.source_path) / 'red_masks' / f'{name}_mask.png', Path(dataset.source_path) / 'red_masks' / f'{name}_mask.jpg'])
        full_mask_path = first_existing([Path(dataset.source_path) / 'masks' / f'{name}_mask.png', Path(dataset.source_path) / 'masks' / f'{name}_mask.jpg'])

        detect_img_chw = rgb # Default
        if red_mask_path and full_mask_path:
            red_mask = torch.from_numpy(np.array(Image.open(red_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
            full_mask = torch.from_numpy(np.array(Image.open(full_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
            red_mask = F.interpolate(red_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
            full_mask = F.interpolate(full_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
            red_mask3 = red_mask.repeat(3, 1, 1)
            full_mask3 = full_mask.repeat(3, 1, 1)

            render_pkg_ori = render(cam, gaussians_original, pipe, bg, iteration=60000, is_train=False)
            rgb_ori = render_pkg_ori['render']
            
            full_adv_car = rgb_ori * (1 - red_mask3) + rgb * red_mask3
            
            relit_image_pil = None
            source_image_path = first_existing([Path(dataset.source_path) / 'ori' / f'{name}.jpg', Path(dataset.source_path) / 'ori' / f'{name}.png'])
            if source_image_path:
                source_image = Image.open(source_image_path).convert('RGB').resize((W, H), Image.BILINEAR)
                if relighter:
                    fg_relight_image = Image.fromarray((rgb_ori.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    bg_mask = Image.open(full_mask_path)
                    relit_image_pil = relighter.relight_hdr(source_image=source_image, fg_relight_image=fg_relight_image, bg_mask=bg_mask, width=W, height=H, invert_mask=True, hdr_sh_coeffs=None)
                else:
                    relit_image_pil = source_image
            
            if relit_image_pil:
                relit_image_tensor = torch.from_numpy(np.array(relit_image_pil)).permute(2,0,1).to(device=rgb.device, dtype=torch.float32) / 255.0
                detect_img_chw = relit_image_tensor * (1 - full_mask3) + full_adv_car * full_mask3
            else:
                detect_img_chw = rgb_ori * (1 - full_mask3) + full_adv_car * full_mask3
        
        save_image_rgb01(detect_img_chw, output_dir / f"{name}.png")

    return output_dir


@torch.no_grad()
def evaluate_from_saved_images(detector, image_dir: Path, anno_dir: Path, args: argparse.Namespace):
    """Evaluates a detector on a directory of pre-rendered images."""
    print(f"    -> 正在评估文件夹: {image_dir.name} ...")

    vis_dir = None
    if bool(getattr(args, 'save_final_eval_vis', False)):
        vis_dir = image_dir.parent / (image_dir.name + '_vis')
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"      -> 检测结果可视化将保存到: {vis_dir}")

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not image_paths:
        return 0.0, 0, 0, 0.0

    successful_attacks = 0
    all_preds_for_map = []
    all_gts_for_map = []
    
    target_class_idx = coco_classes.index(args.target_class_name)

    # Process images in batches to use inference_detector_custom
    batch_size = 8  # A reasonable batch size
    path_batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    pbar = tqdm(path_batches, desc=f"评估 {detector.__class__.__name__} on {image_dir.name}", ncols=120, leave=False)
    for batch_paths in pbar:
        imgs_for_det = []
        view_names_with_anno = []
        gt_bboxes_batch = []
        gt_labels_batch = []

        for img_path in batch_paths:
            anno_path = anno_dir / f'{img_path.stem}.json'
            if not anno_path.exists():
                continue
            
            gt_bboxes, gt_label_name = load_labelme_annotation(str(anno_path))
            if gt_bboxes is None:
                continue
            
            try:
                gt_label_idx = coco_classes.index(gt_label_name)
            except ValueError:
                continue

            gt_bboxes_batch.append(gt_bboxes)
            gt_labels_batch.append(np.array([gt_label_idx] * len(gt_bboxes)))
            all_gts_for_map.append({'bboxes': gt_bboxes, 'labels': np.array([gt_label_idx] * len(gt_bboxes))})

            img_np = np.array(Image.open(img_path).convert('RGB'))
            img_tensor = torch.from_numpy(img_np).to(args.device) # Correct: (H, W, C) tensor
            imgs_for_det.append(img_tensor)
            view_names_with_anno.append(img_path.stem)

        if not imgs_for_det:
            continue
        
        imgs_for_det_batch = torch.stack(imgs_for_det, dim=0) # Correct: (B, H, W, C) tensor

        # Use inference_detector_custom for batch processing and visualization
        preds = inference_detector_custom(
            detector, imgs_for_det_batch, gt_bboxes_list=None, 
            vis_dir=vis_dir, view_names=view_names_with_anno
        )

        for i, pred in enumerate(preds):
            pred_instances = pred.pred_instances
            
            score_thresh = getattr(args, 'score_thresh', 0.5)
            score_mask = pred_instances.scores >= score_thresh
            class_mask = pred_instances.labels == target_class_idx
            match_indices = (score_mask & class_mask).nonzero(as_tuple=False).squeeze(1)
            
            is_attack_successful = True
            if match_indices.numel() > 0:
                pred_bboxes_tc = pred_instances.bboxes[match_indices]
                
                current_gt_bboxes = gt_bboxes_batch[i]
                current_gt_labels = gt_labels_batch[i]
                
                if current_gt_bboxes is not None and len(current_gt_bboxes) > 0:
                    sel = (current_gt_labels == target_class_idx)
                    gt_sel_np = current_gt_bboxes[sel] if sel.any() else np.array([])
                    
                    if gt_sel_np.shape[0] > 0:
                        gt_sel_t = torch.from_numpy(gt_sel_np).to(pred_bboxes_tc.device, dtype=pred_bboxes_tc.dtype)
                        max_iou = torch.zeros(pred_bboxes_tc.shape[0], device=pred_bboxes_tc.device)
                        for g in gt_sel_t:
                            ious = compute_iou(pred_bboxes_tc, g.unsqueeze(0))
                            max_iou = torch.maximum(max_iou, ious)
                        if (max_iou >= 0.5).any():
                            is_attack_successful = False
            
            if is_attack_successful:
                successful_attacks += 1
                
            num_classes = len(detector.CLASSES)
            pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
            for j in range(num_classes):
                class_indices = (pred.pred_instances.labels == j)
                if class_indices.any():
                    boxes = pred.pred_instances.bboxes[class_indices].cpu().numpy()
                    scores = pred.pred_instances.scores[class_indices].cpu().numpy()
                    pred_for_map[j] = np.hstack([boxes, scores[:, np.newaxis]])
            all_preds_for_map.append(pred_for_map)

    total_attacks = len(all_gts_for_map)
    if total_attacks == 0:
        return 0.0, 0, 0, 0.0

    final_asr = successful_attacks / total_attacks
    eval_results = calculate_ap_for_target_class(all_preds_for_map, all_gts_for_map, target_class_idx, iou_thr=0.5)
    ap50 = eval_results.get('AP50', 0.0)
    
    return final_asr, successful_attacks, total_attacks, ap50


def load_hdr_image_and_tonemap(hdr_path, gamma=2.2):
    """Loads and tonemaps an HDR image using OpenCV, returning a PIL Image."""
    try:
        hdr_cv = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr_cv is None:
            print(f"无法加载HDR文件: {hdr_path}")
            return None
        
        hdr_cv = cv2.cvtColor(hdr_cv, cv2.COLOR_BGR2RGB)

        if hdr_cv.dtype != np.float32:
            hdr_cv = hdr_cv.astype(np.float32)

        luminance = 0.2126 * hdr_cv[..., 0] + 0.7152 * hdr_cv[..., 1] + 0.0722 * hdr_cv[..., 2]
        scale = 1.0 / (1.0 + luminance[..., np.newaxis] + 1e-6)
        ldr_cv = hdr_cv * scale

        ldr_cv = np.power(ldr_cv, 1.0/gamma)

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


@torch.no_grad()
def render_and_save_final_images_mw(cameras, gaussians, pipe, bg, args, dataset, gaussians_original, relighter, save_dir: Path, set_name: str):
	"""
	多天气渲染与保存：使用 ori_mw2 下不同天气的同视角背景，并为每种天气加载对应的 HDR 环境贴图，
	使前景渲染（车辆）在该天气的光照下生成，然后与该天气背景拼接。

	优先使用：
	- 背景来源：dataset.source_path / 'ori_mw2' / <weather> / 'XX_<weather>.png'
	- HDR 来源：dataset.source_path / '_EnvironmentMaps' / <...weather...>.hdr|.exr
	若 ori_mw2 不存在则回退到 ori_mw（仅背景），HDR 仍尝试从 _EnvironmentMaps 加载。

	规则：
	- 其中 XX 为去掉末尾一次下划线段后的前缀（从原始 name 中去掉原来的天气后缀）。
	- 输出：为每个天气分别创建一个输出目录，文件名仍保持原始 name（便于与 annos 匹配）。
	返回：
	- 一个字典 { weather_name: output_dir_path }
	"""
	# Prefer new dataset folder layout
	ori_mw_root = Path(dataset.source_path) / 'ori_mw2'
	if not ori_mw_root.is_dir():
		ori_mw_root_fallback = Path(dataset.source_path) / 'ori_mw'
		if ori_mw_root_fallback.is_dir():
			ori_mw_root = ori_mw_root_fallback
			print(f"[消息] [MW] 未找到 ori_mw2，回退使用: {ori_mw_root}")
		else:
			print(f"[消息] [MW] 未找到多天气目录: {ori_mw_root} 或 {ori_mw_root_fallback}，跳过跨光渲染。")
			return {}

	envmaps_root = Path(dataset.source_path) / 'ori_mw2' / '_EnvironmentMaps'
	if not envmaps_root.is_dir():
		print(f"[警告] [MW] 未找到环境贴图目录: {envmaps_root}。将继续渲染，但前景将使用当前 envlight 光照（不做天气对应 HDR 切换）。")

	# 收集天气列表（子文件夹）
	# 注意：跳过 _EnvironmentMaps / __EnvironmentMaps 等非天气目录，避免生成 final_full_images__EnvironmentMaps
	weather_dirs = sorted([
		d for d in ori_mw_root.iterdir()
		if d.is_dir() and (not d.name.startswith('_')) and ('EnvironmentMaps' not in d.name)
	])
	if not weather_dirs:
		print(f"[消息] [MW] 目录 '{ori_mw_root}' 下无天气子文件夹，跳过跨光渲染。")
		return {}

	def _match_mw_path(weather_dir: Path, name: str, weather_name: str) -> Path | None:
		# name 去掉末尾一次 '_' 段：ori_xxx_sunny -> ori_xxx
		prefix = name.rsplit('_', 1)[0] if '_' in name else name
		candidate = weather_dir / f"{prefix}_{weather_name}.png"
		if candidate.is_file():
			return candidate
		# 回退：尝试 jpg
		candidate_jpg = weather_dir / f"{prefix}_{weather_name}.jpg"
		if candidate_jpg.is_file():
			return candidate_jpg
		# 最后回退：遍历匹配以 prefix 开头且以 _{weather_name} 结尾的 png/jpg
		for p in sorted(weather_dir.iterdir()):
			if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
				stem = p.stem
				if stem.endswith(f"_{weather_name}") and stem.startswith(prefix):
					return p
		return None

	def _find_weather_hdr(env_root: Path, weather_name: str) -> Path | None:
		"""
		Try to locate a per-weather HDR/EXR file under env_root.
		Heuristics:
		- exact stem == weather_name
		- stem contains weather_name (case-insensitive)
		- filename contains weather_name
		"""
		if env_root is None or (not Path(env_root).is_dir()):
			return None
		w = str(weather_name).lower()
		candidates = []
		for ext in ('*.hdr', '*.exr', '*.HDR', '*.EXR'):
			candidates.extend(list(Path(env_root).rglob(ext)))
		if not candidates:
			return None
		# 1) exact stem match
		for p in candidates:
			if p.stem.lower() == w:
				return p
		# 2) stem contains weather
		for p in candidates:
			if w in p.stem.lower():
				return p
		# 3) fallback: filename contains weather
		for p in candidates:
			if w in p.name.lower():
				return p
		# 4) deterministic fallback: first candidate sorted by name
		try:
			return sorted(candidates, key=lambda x: x.name.lower())[0]
		except Exception:
			return candidates[0]

	results = {}

	# Preserve current envlight base to avoid side effects
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

	for wdir in weather_dirs:
		weather_name = wdir.name
		output_dir = save_dir / f"final_{set_name}_images_{weather_name}"
		output_dir.mkdir(parents=True, exist_ok=True)
		print(f"\n[消息] [MW] 开始渲染并保存 '{set_name}' 集的多天气='{weather_name}' 图像到: {output_dir}")

		# Load per-weather environment map (HDR) if available; affects foreground rendering lighting.
		hdr_path = _find_weather_hdr(envmaps_root, weather_name) if envmaps_root.is_dir() else None
		if hdr_path is not None and Path(hdr_path).is_file():
			try:
				gaussians.envlight.scale = getattr(args, 'environment_scale', 1.0)
				gaussians.envlight.load(str(hdr_path))
				gaussians.envlight.build_mips()
				gaussians_original.envlight.scale = getattr(args, 'environment_scale', 1.0)
				gaussians_original.envlight.load(str(hdr_path))
				gaussians_original.envlight.build_mips()
				print(f"[消息] [MW] 已加载天气 '{weather_name}' 的 HDR: {hdr_path.name}")
			except Exception as e:
				print(f"[警告] [MW] 加载/构建天气 HDR 失败 ({weather_name}): {hdr_path}: {e}")
		else:
			if envmaps_root.is_dir():
				print(f"[警告] [MW] 未找到天气 '{weather_name}' 的 HDR/EXR 于: {envmaps_root}。将使用当前 envlight 光照渲染前景。")

		for cam in tqdm(cameras, desc=f"渲染 {set_name} 集(多天气={weather_name})", ncols=120):
			name = cam.image_name
			anno_path = Path(dataset.source_path) / 'annos' / f'{name}.json'
			if not anno_path.exists():
				continue

			render_pkg = render(
				cam, gaussians, pipe, bg, iteration=60000, is_train=False,
				second_stage_step=int(args.second_stage_step), hdr_rotation=bool(args.hdr_rotation)
			)
			rgb = render_pkg['render']
			H, W = rgb.shape[-2], rgb.shape[-1]

			red_mask_path = first_existing([Path(dataset.source_path) / 'red_masks' / f'{name}_mask.png', Path(dataset.source_path) / 'red_masks' / f'{name}_mask.jpg'])
			full_mask_path = first_existing([Path(dataset.source_path) / 'masks' / f'{name}_mask.png', Path(dataset.source_path) / 'masks' / f'{name}_mask.jpg'])

			detect_img_chw = rgb  # default
			if red_mask_path and full_mask_path:
				red_mask = torch.from_numpy(np.array(Image.open(red_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
				full_mask = torch.from_numpy(np.array(Image.open(full_mask_path).convert('L'))).to(device=rgb.device, dtype=torch.float32) / 255.0
				red_mask = F.interpolate(red_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
				full_mask = F.interpolate(full_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
				red_mask3 = red_mask.repeat(3, 1, 1)
				full_mask3 = full_mask.repeat(3, 1, 1)

				render_pkg_ori = render(cam, gaussians_original, pipe, bg, iteration=60000, is_train=False)
				rgb_ori = render_pkg_ori['render']
				full_adv_car = rgb_ori * (1 - red_mask3) + rgb * red_mask3

				# 取多天气背景
				mw_path = _match_mw_path(wdir, name, weather_name)
				if mw_path and Path(mw_path).is_file():
					try:
						mw_bg = Image.open(mw_path).convert('RGB').resize((W, H), Image.BILINEAR)
						mw_bg_tensor = torch.from_numpy(np.array(mw_bg)).permute(2, 0, 1).to(device=rgb.device, dtype=torch.float32) / 255.0
						detect_img_chw = mw_bg_tensor * (1 - full_mask3) + full_adv_car * full_mask3
					except Exception as e:
						print(f"[警告] [MW] 加载/处理背景失败: {mw_path.name}: {e}")
						detect_img_chw = full_adv_car  # 退化为白底合成
				else:
					# 未找到对应多天气背景，退化为白底合成
					detect_img_chw = full_adv_car

			save_image_rgb01(detect_img_chw, output_dir / f"{name}.png")

		results[weather_name] = output_dir

	# Restore previous envlight base
	try:
		if prev_gauss_base_cpu is not None:
			gaussians.envlight.base = prev_gauss_base_cpu.to(device=bg.device if hasattr(bg, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
			gaussians.envlight.build_mips()
	except Exception:
		pass
	try:
		if prev_orig_base_cpu is not None:
			gaussians_original.envlight.base = prev_orig_base_cpu.to(device=bg.device if hasattr(bg, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
			gaussians_original.envlight.build_mips()
	except Exception:
		pass

	return results


@torch.no_grad()
def visualize_hdr_bank_from_dir(cameras: List, gaussians, pipe, bg: torch.Tensor, args: argparse.Namespace, dataset: ModelParams, hdr_bank_dir: Path, save_root: Path, num_views: int = 5, seed: int = 0):
	"""
	从目录遍历 HDR/EXR 文件，随机抽取 num_views 个视角，对每个 HDR 进行渲染可视化并保存。
	base 来源：envlight.load(hdr)；随后仅 build_mips()，不调用 build_base()。
	保存结构：save_root/<hdr_stem>/<camera_name>.png
	"""
	hdr_dir = Path(hdr_bank_dir)
	if not hdr_dir.is_dir():
		print(f"[警告] [HDR-VIS] 目录不存在: {hdr_dir}")
		return None

	hdr_files = sorted([p for p in hdr_dir.iterdir() if p.suffix.lower() in ['.hdr', '.exr']])
	if not hdr_files:
		print(f"[警告] [HDR-VIS] 未在 {hdr_dir} 找到 .hdr/.exr 文件")
		return None

	save_root = Path(save_root)
	save_root.mkdir(parents=True, exist_ok=True)

	# 采样视角
	random.seed(seed)
	cams = list(cameras)
	if not cams:
		print("[警告] [HDR-VIS] 提供的相机列表为空，跳过可视化。")
		return None
	sampled = random.sample(cams, min(num_views, len(cams)))

	print(f"[消息] [HDR-VIS] 将对 {len(hdr_files)} 个HDR、{len(sampled)} 个视角进行渲染，可视化保存到: {save_root}")

	for hdr_path in tqdm(hdr_files, desc="HDR-VIS (dir)", ncols=120):
		subdir = save_root / hdr_path.stem
		subdir.mkdir(parents=True, exist_ok=True)
		try:
			gaussians.envlight.scale = getattr(args, 'environment_scale', 1.0)
			gaussians.envlight.load(str(hdr_path))
			gaussians.envlight.build_mips()
		except Exception as e:
			print(f"[警告] [HDR-VIS] 加载/构建 mips 失败: {hdr_path.name}: {e}")
			continue

		for cam in sampled:
			try:
				render_pkg = render(
					cam, gaussians, pipe, bg,
					iteration=60000,
					is_train=False,
					second_stage_step=int(getattr(args, 'second_stage_step', 30000)),
					hdr_rotation=bool(getattr(args, 'hdr_rotation', False))
				)
				rgb = render_pkg['render']
				save_image_rgb01(rgb, subdir / f"{cam.image_name}.png")
			except Exception as e:
				print(f"[警告] [HDR-VIS] 渲染失败: {hdr_path.stem}/{cam.image_name}: {e}")

	return save_root


@torch.no_grad()
def visualize_hdr_bases_with_random_views(cameras: List, gaussians, pipe, bg: torch.Tensor, args: argparse.Namespace, dataset: ModelParams, hdr_bases: List, save_root: Path, num_views: int = 5, seed: int = 0):
	"""
	基于内存中的 envlight.base 列表进行可视化（如 ReplayBuffer 中的 base）。
	hdr_bases: List[Tuple[str, torch.Tensor]] 或 List[torch.Tensor]，若为张量则自动命名 base_XXX。
	base 直接赋给 gaussians.envlight.base；随后仅 build_mips()，不调用 build_base()。
	保存结构：save_root/<name>/<camera_name>.png
	"""
	if not hdr_bases:
		print("[警告] [HDR-VIS] 提供的 hdr_bases 为空，跳过可视化。")
		return None

	save_root = Path(save_root)
	save_root.mkdir(parents=True, exist_ok=True)

	# 采样视角
	random.seed(seed)
	cams = list(cameras)
	if not cams:
		print("[警告] [HDR-VIS] 提供的相机列表为空，跳过可视化。")
		return None
	sampled = random.sample(cams, min(num_views, len(cams)))

	# 规范化输入 (name, tensor)
	named_bases = []
	for idx, item in enumerate(hdr_bases):
		if isinstance(item, tuple) and len(item) == 2:
			nm, bt = item
			named_bases.append((str(nm), bt))
		else:
			named_bases.append((f"base_{idx:03d}", item))

	print(f"[消息] [HDR-VIS] 将对 {len(named_bases)} 个base、{len(sampled)} 个视角进行渲染，可视化保存到: {save_root}")

	device = bg.device if hasattr(bg, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for name, base in tqdm(named_bases, desc="HDR-VIS (bases)", ncols=120):
		subdir = save_root / name
		subdir.mkdir(parents=True, exist_ok=True)
		try:
			gaussians.envlight.base = base.to(device=device, dtype=torch.float32)
			gaussians.envlight.build_mips()
		except Exception as e:
			print(f"[警告] [HDR-VIS] 设置 base/构建 mips 失败: {name}: {e}")
			continue

		for cam in sampled:
			try:
				render_pkg = render(
					cam, gaussians, pipe, bg,
					iteration=60000,
					is_train=False,
					second_stage_step=int(getattr(args, 'second_stage_step', 30000)),
					hdr_rotation=bool(getattr(args, 'hdr_rotation', False))
				)
				rgb = render_pkg['render']
				save_image_rgb01(rgb, subdir / f"{cam.image_name}.png")
			except Exception as e:
				print(f"[警告] [HDR-VIS] 渲染失败: {name}/{cam.image_name}: {e}")

	return save_root

