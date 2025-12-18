from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch


def _load_mmdet(cfg_path: str, checkpoint: str, device: str = "cuda"):
	# Lazy import to avoid hard dependency if user doesn't run this path
	from mmengine.config import Config
	from mmdet.apis import init_detector

	cfg = Config.fromfile(cfg_path)
	model = init_detector(cfg, checkpoint, device=device)
	# Ensure we don't accidentally swap color spaces; our renders are RGB
	try:
		if hasattr(model, 'data_preprocessor') and hasattr(model.data_preprocessor, 'bgr_to_rgb'):
			model.data_preprocessor.bgr_to_rgb = False
	except Exception:
		pass
	# Use training graph for differentiability but keep weights frozen
	model.train()
	for p in model.parameters():
		p.requires_grad_(False)
	return model, cfg


def _normalize_for_mmdet(img: torch.Tensor, mean: List[float], std: List[float], to_bgr: bool = True) -> torch.Tensor:
	"""
	img: (3,H,W) in [0,1] RGB, float32, CUDA
	return: normalized tensor as mmdet backbone expects
	"""
	assert img.dim() == 3 and img.shape[0] == 3
	out = img * 255.0
	if to_bgr:
		out = out[[2, 1, 0], ...]
	mean_t = torch.tensor(mean, device=out.device).view(3, 1, 1)
	std_t = torch.tensor(std, device=out.device).view(3, 1, 1)
	return (out - mean_t) / std_t


class CocoGtLookup:
	def __init__(self, coco_json_path: Optional[str]) -> None:
		self.name_to_instances: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[List[Any]]]] = {}
		if coco_json_path is None:
			return
		data = json.loads(Path(coco_json_path).read_text())
		# Build category id to label idx mapping (contiguous)
		cat_ids = [c['id'] for c in data['categories']]
		cat_ids_sorted = sorted(cat_ids)
		catid_to_label = {cid: i for i, cid in enumerate(cat_ids_sorted)}
		# index images by file_name (stem without extension)
		id_to_name: Dict[int, str] = {img['id']: Path(img['file_name']).stem for img in data['images']}
		from collections import defaultdict
		name_to_boxes: Dict[str, List[List[float]]] = defaultdict(list)
		name_to_labels: Dict[str, List[int]] = defaultdict(list)
		for ann in data['annotations']:
			img_id = ann['image_id']
			name = id_to_name.get(img_id, None)
			if name is None:
				continue
			bbox = ann['bbox']  # [x,y,w,h]
			# convert to xyxy
			x, y, w, h = bbox
			name_to_boxes[name].append([x, y, x + w, y + h])
			name_to_labels[name].append(catid_to_label[ann['category_id']])
		for name, boxes in name_to_boxes.items():
			labels = name_to_labels[name]
			self.name_to_instances[name] = (
				torch.tensor(boxes, dtype=torch.float32, device='cuda') if boxes else torch.zeros((0, 4), device='cuda'),
				torch.tensor(labels, dtype=torch.long, device='cuda') if labels else torch.zeros((0,), dtype=torch.long, device='cuda'),
				none_if_empty(labels)
			)

	def get(self, name_stem: str) -> Tuple[torch.Tensor, torch.Tensor]:
		b, l, _ = self.name_to_instances.get(name_stem, (torch.zeros((0,4), device='cuda'), torch.zeros((0,), dtype=torch.long, device='cuda'), None))
		return b, l


def none_if_empty(x: List[Any]) -> Optional[List[Any]]:
	return None if len(x) == 0 else x



class MMDetLoss:
	def __init__(self, cfg_path: str, checkpoint: str, device: str = 'cuda', mean: List[float] | None = None, std: List[float] | None = None, to_bgr: bool = True) -> None:
		self.model, self.cfg = _load_mmdet(cfg_path, checkpoint, device)
		# Keep legacy fields for potential downstream usage, but not used with data_preprocessor
		if mean is None or std is None:
			self.mean = [123.675, 116.28, 103.53]
			self.std = [58.395, 57.12, 57.375]
		else:
			self.mean = mean
			self.std = std
		self.to_bgr = to_bgr

	def loss(self, img_name_stem: str, render_rgb01: torch.Tensor, gt_lookup: CocoGtLookup) -> torch.Tensor:
		"""Compute detection loss for single image using MMDet v3 API.
		- render_rgb01: (3,H,W) in [0,1], RGB, CUDA
		- returns scalar loss tensor
		"""
		# Prepare inputs as raw image tensor in [0,255], letting model.data_preprocessor handle normalization
		img = (render_rgb01 * 255.0).unsqueeze(0).contiguous()  # (1,3,H,W)
		H, W = int(render_rgb01.shape[1]), int(render_rgb01.shape[2])
		# Ground truth instances
		gt_bboxes, gt_labels = gt_lookup.get(img_name_stem)
		gt_bboxes = gt_bboxes.to(img.device)
		gt_labels = gt_labels.to(img.device)
		# Build data sample for v3 loss
		from mmdet.structures import DetDataSample
		from mmengine.structures import InstanceData
		inst = InstanceData()
		inst.bboxes = gt_bboxes
		inst.labels = gt_labels
		ds = DetDataSample()
		ds.set_metainfo({
			'img_shape': (H, W),
			'ori_shape': (H, W),
			'pad_shape': (H, W),
			'batch_input_shape': (H, W),
			'scale_factor': 1.0,
		})
		ds.gt_instances = inst
		# Compute losses and reduce to scalar
		loss_dict = self.model.loss(img, [ds])
		device = img.device
		def reduce_to_scalar(x):
			if torch.is_tensor(x):
				return x.sum()
			if isinstance(x, (list, tuple)):
				return sum(reduce_to_scalar(i) for i in x)
			if isinstance(x, dict):
				return sum(reduce_to_scalar(v) for v in x.values())
			return torch.as_tensor(x, device=device).sum()
		if isinstance(loss_dict, dict):
			loss = reduce_to_scalar(loss_dict)
		else:
			loss = reduce_to_scalar(loss_dict)
		# ensure 0-dim tensor
		if not torch.is_tensor(loss):
			loss = torch.as_tensor(loss, device=device)
		return loss


