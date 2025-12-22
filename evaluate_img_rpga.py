# -*- coding: utf-8 -*-
"""
评估 RPGA/RGA 输出目录下的“最终多天气渲染结果”。

与 `evaluate_img_mw2.py` 的主要区别：
- 输入不是 EXP_EVAL 下的 *_mw2 方法文件夹
- 而是给定一个运行输出目录，例如：RGA_output/1218_191840_Beijing
- 脚本会自动发现其下的多天气结果文件夹，例如：
    final_full_images_Dark/
    final_full_images_Foggy/
    ...
  并对这些天气文件夹中的图片进行评估，最后输出中文汇总 txt。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from utils.main_utils import (
	calculate_ap_for_target_class,
	coco_classes,
	load_labelme_annotation,
	compute_iou,
)


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
	'pvt': {
		'config': 'configs/pvt/retinanet_pvt-m_fpn_1x_coco.py',
		'ckpt': 'checkpoints/retinanet_pvt-m_fpn_1x_coco.pth'
	},
	'detr': {
		'config': 'configs/detr/detr_r50_8xb2-150e_coco.py',
		'ckpt': 'checkpoints/detr_r50_8xb2-150e_coco.pth'
	}
}

IMG_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def parse_args():
	parser = argparse.ArgumentParser(
		description="评估单个 RGA_output/<timestamp> 目录下的 final_*_images_* 多天气结果，并输出 pitch/distance/weather/detector 四种对比汇总表（txt）"
	)
	parser.add_argument(
		'--exp_dir',
		default='./RGA_output/1220_195202_Beijing',
		help='运行输出目录（例如 RGA_output/1220_195003_Beijing），其下包含 final_full_images_Dark 等文件夹'
	)
	parser.add_argument(
		'--anno_dir',
		default='./data/carla_full_sunny/annos',
		help='包含 LabelMe 标注 (.json) 的目录路径（通常是 data/.../annos）。若为空将尝试使用 <exp_dir>/../annos（不一定存在）。'
	)
	parser.add_argument(
		'--mmdet_base',
		default='./mmdet_files',
		help='mmdet_files 根目录（包含 configs/ 与 checkpoints/）'
	)
	parser.add_argument(
		'--detector',
		type=str,
		default='detr',
		choices=list(DETECTOR_PATHS.keys()),
		help='单个检测器（当不启用 --all_detectors 时使用）'
	)
	parser.add_argument(
		'--all_detectors',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='是否评估所有检测器（默认开启）。若关闭则仅评估 --detector（可用 --no-all_detectors）'
	)
	parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for ASR calculation.')
	parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for considering a detection valid.')
	parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
	parser.add_argument(
		'--output_file',
		default='',
		help='输出 txt 文件名/路径（中文汇总）。若为空则输出到 <exp_dir>/evaluation_results_rpga.txt'
	)
	parser.add_argument('--skip_suffix', default='_vis', help='跳过以该后缀结尾的结果文件夹（默认 _vis，用于忽略可视化目录）。设置为空则不跳过。')
	return parser.parse_args()


def _safe_int_from_regex(stem: str, pattern: str) -> int | None:
	m = re.search(pattern, stem)
	if not m:
		return None
	try:
		return int(m.group(1))
	except Exception:
		return None


def _parse_pitch_distance(stem: str) -> tuple[int | None, int | None]:
	pitch = _safe_int_from_regex(stem, r'(?i)pitch(\d+)')
	distance = _safe_int_from_regex(stem, r'(?i)distance(\d+)')
	return pitch, distance


def _init_one_detector(detector_name: str, device: str, mmdet_base: Path):
	print(f"[INFO] 初始化检测器：{detector_name} ...")
	selected = DETECTOR_PATHS.get(detector_name)
	if selected is None:
		raise ValueError(f"不支持的检测器：{detector_name}")
	cfg_path = str(mmdet_base / selected['config'])
	ckpt_path = str(mmdet_base / selected['ckpt'])
	detector = init_detector(cfg_path, ckpt_path, device=device)
	if not hasattr(detector, 'CLASSES'):
		detector.CLASSES = coco_classes
	return detector


def _discover_weather_dirs(exp_dir: Path, skip_suffix: str) -> list[tuple[str, Path]]:
	"""
	发现 exp_dir 下的 final_*_images_* 目录，返回 [(weather_name, dir_path), ...]
	"""
	if not exp_dir.exists():
		return []
	out = []
	for p in exp_dir.iterdir():
		if not p.is_dir():
			continue
		if skip_suffix and p.name.endswith(skip_suffix):
			continue
		# match: final_<split>_images_<weather>
		m = re.match(r'^final_(?P<split>.+)_images_(?P<weather>.+)$', p.name)
		if not m:
			continue
		weather = m.group('weather')
		# Ignore special/internal folders if any (e.g., "__EnvironmentMaps" used for debugging)
		if weather.startswith('_') or 'EnvironmentMaps' in weather:
			continue
		out.append((weather, p))
	return sorted(out, key=lambda x: x[0].lower())


def _match_annotation_path(img_path: Path, anno_files: list[Path], anno_dir: Path, weather_name: str | None) -> Path | None:
	"""
	与 evaluate_img_mw2.py 保持一致的“宽松匹配”：
	- 若文件名中出现 weather_name，则取其前缀作为标注前缀
	- 否则回退：去掉最后一个 '_' 段的前缀
	然后在 anno_files 中找 stem 以该前缀开头的最短匹配。
	"""
	stem = img_path.stem
	prefix = None
	if weather_name:
		idx = stem.find(str(weather_name))
		if idx != -1:
			prefix = stem[:idx].rstrip('_-.')
	if not prefix:
		if '_' in stem:
			prefix = stem.rsplit('_', 1)[0]
		else:
			prefix = stem

	matched = [p for p in anno_files if p.stem.startswith(prefix)]
	if matched:
		return min(matched, key=lambda p: len(p.stem))
	fallback = anno_dir / f'{stem}.json'
	return fallback if fallback.exists() else None


def _evaluate_weather_dir_collect(
	detector,
	detector_name: str,
	method_name: str,
	weather_dir: Path,
	weather_name: str,
	anno_dir: Path,
	anno_files: list[Path],
	target_class_name: str,
	score_thresh: float,
) -> tuple[list[dict], list[list[np.ndarray]], list[dict], int]:
	try:
		target_class_idx = coco_classes.index(target_class_name)
	except ValueError:
		raise ValueError(f"目标类别不在 COCO classes 中：{target_class_name}")

	image_paths = sorted([p for p in weather_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES])
	if not image_paths:
		return [], [], [], target_class_idx

	local_records: list[dict] = []
	local_preds_for_map: list[list[np.ndarray]] = []
	local_gts_for_map: list[dict] = []

	num_classes = len(detector.CLASSES)

	for img_path in tqdm(image_paths, ncols=100, desc=f"{method_name}/{weather_name}/{detector_name}"):
		anno_path = _match_annotation_path(img_path, anno_files, anno_dir, weather_name)
		if anno_path is None:
			continue

		gt_bboxes, gt_label_name = load_labelme_annotation(str(anno_path))
		if gt_bboxes is None:
			continue

		try:
			gt_label_idx = coco_classes.index(gt_label_name)
		except ValueError:
			continue

		# 只统计目标类别样本（保持 ASR/AP 语义一致）
		if gt_label_idx != target_class_idx:
			continue

		# inference
		try:
			img_np = np.array(Image.open(img_path).convert('RGB'))
			result = inference_detector(detector, img_np)
			pred_instances = result.pred_instances
		except Exception as e:
			print(f"\n[WARNING] 处理失败：{img_path.name}，原因：{e}")
			continue

		# ASR：攻击成功当且仅当不存在“score>=阈值 且 与任一GT IoU>=0.5”的目标类预测
		is_attack_successful = True
		score_mask = pred_instances.scores >= score_thresh
		class_mask = pred_instances.labels == target_class_idx
		match_indices = (score_mask & class_mask).nonzero(as_tuple=False).squeeze(1)
		if match_indices.numel() > 0 and gt_bboxes is not None and len(gt_bboxes) > 0:
			pred_bboxes_tc = pred_instances.bboxes[match_indices]  # [K,4]
			gt_t = torch.from_numpy(gt_bboxes).to(pred_bboxes_tc.device, dtype=pred_bboxes_tc.dtype)
			max_iou = torch.zeros(pred_bboxes_tc.shape[0], device=pred_bboxes_tc.device)
			for g in gt_t:
				ious = compute_iou(pred_bboxes_tc, g.unsqueeze(0))  # [K]
				max_iou = torch.maximum(max_iou, ious)
			if (max_iou >= 0.5).any():
				is_attack_successful = False

		pitch, distance = _parse_pitch_distance(img_path.stem)

		local_records.append({
			'method': method_name,
			'weather': weather_name,
			'detector': detector_name,
			'image_name': img_path.name,
			'pitch': pitch,
			'distance': distance,
			'attack_successful': bool(is_attack_successful),
		})

		local_gts_for_map.append({
			'bboxes': gt_bboxes,
			'labels': np.array([gt_label_idx] * len(gt_bboxes))
		})

		pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
		for i in range(num_classes):
			class_indices = (pred_instances.labels == i)
			if class_indices.any():
				boxes = pred_instances.bboxes[class_indices].cpu().numpy()
				scores = pred_instances.scores[class_indices].cpu().numpy()
				pred_for_map[i] = np.hstack([boxes, scores[:, np.newaxis]])
		local_preds_for_map.append(pred_for_map)

	return local_records, local_preds_for_map, local_gts_for_map, target_class_idx


def _compute_metrics_for_indices(
	indices: list[int],
	records: list[dict],
	preds_for_map: list[list[np.ndarray]],
	gts_for_map: list[dict],
	target_class_idx: int,
) -> tuple[float, float, int, int]:
	n = len(indices)
	if n == 0:
		return float('nan'), float('nan'), 0, 0
	succ = sum(1 for i in indices if records[i]['attack_successful'])
	asr = succ / n
	sub_preds = [preds_for_map[i] for i in indices]
	sub_gts = [gts_for_map[i] for i in indices]
	ap50 = calculate_ap_for_target_class(sub_preds, sub_gts, target_class_idx, iou_thr=0.5).get('AP50', float('nan'))
	return asr, ap50, n, succ


def _write_txt_report(
	output_file: Path,
	records: list[dict],
	preds_for_map: list[list[np.ndarray]],
	gts_for_map: list[dict],
	target_class_idx: int,
):
	methods = sorted({r['method'] for r in records})
	all_pitches = sorted({r['pitch'] for r in records if r.get('pitch') is not None})
	all_distances = sorted({r['distance'] for r in records if r.get('distance') is not None})
	all_weathers = sorted({r['weather'] for r in records})
	all_detectors = sorted({r['detector'] for r in records})

	def idxs_where(**kwargs):
		out = []
		for i, r in enumerate(records):
			ok = True
			for k, v in kwargs.items():
				if r.get(k) != v:
					ok = False
					break
			if ok:
				out.append(i)
		return out

	lines: list[str] = []
	lines.append("========== RPGA/RGA 多天气评估汇总（中文 txt）==========")
	lines.append(f"总样本数（逐图统计）：{len(records)}")
	lines.append(f"方法数量：{len(methods)}，天气数量：{len(all_weathers)}，距离种类：{len(all_distances)}，俯仰角种类：{len(all_pitches)}，检测器数量：{len(all_detectors)}")
	lines.append("")

	# 1) pitch
	lines.append("========== 表 1：控制变量【俯仰角 pitch】==========")
	for m in methods:
		lines.append(f"【方法：{m}】")
		for p in all_pitches:
			idxs = [i for i in idxs_where(method=m) if records[i].get('pitch') == p]
			asr, ap50, n, succ = _compute_metrics_for_indices(idxs, records, preds_for_map, gts_for_map, target_class_idx)
			if n == 0:
				continue
			lines.append(f"变量俯仰角：pitch{p}，{m}方法的平均ASR是{asr:.4f}，平均AP@0.5是{ap50:.4f}（成功{succ}/{n}）")
		lines.append("")

	# 2) distance
	lines.append("========== 表 2：控制变量【距离 distance】==========")
	for m in methods:
		lines.append(f"【方法：{m}】")
		for d in all_distances:
			idxs = [i for i in idxs_where(method=m) if records[i].get('distance') == d]
			asr, ap50, n, succ = _compute_metrics_for_indices(idxs, records, preds_for_map, gts_for_map, target_class_idx)
			if n == 0:
				continue
			lines.append(f"变量距离：distance{d}，{m}方法的平均ASR是{asr:.4f}，平均AP@0.5是{ap50:.4f}（成功{succ}/{n}）")
		lines.append("")

	# 3) weather
	lines.append("========== 表 3：控制变量【天气 weather】==========")
	for m in methods:
		lines.append(f"【方法：{m}】")
		for w in all_weathers:
			idxs = idxs_where(method=m, weather=w)
			asr, ap50, n, succ = _compute_metrics_for_indices(idxs, records, preds_for_map, gts_for_map, target_class_idx)
			if n == 0:
				continue
			lines.append(f"变量天气：{w}，{m}方法的平均ASR是{asr:.4f}，平均AP@0.5是{ap50:.4f}（成功{succ}/{n}）")
		lines.append("")

	# 4) detector
	lines.append("========== 表 4：控制变量【检测器 detector】==========")
	for m in methods:
		lines.append(f"【方法：{m}】")
		for det in all_detectors:
			idxs = idxs_where(method=m, detector=det)
			asr, ap50, n, succ = _compute_metrics_for_indices(idxs, records, preds_for_map, gts_for_map, target_class_idx)
			if n == 0:
				continue
			lines.append(f"变量检测器：{det}，{m}方法的平均ASR是{asr:.4f}，平均AP@0.5是{ap50:.4f}（成功{succ}/{n}）")
		lines.append("")

	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text("\n".join(lines), encoding='utf-8')
	print(f"[INFO] 已写出中文汇总到：{output_file}")


def main():
	args = parse_args()

	exp_dir = Path(args.exp_dir)
	if not exp_dir.exists():
		print(f"[ERROR] exp_dir 不存在：{exp_dir}")
		return

	# Resolve anno_dir
	anno_dir = Path(args.anno_dir) if str(args.anno_dir).strip() else None
	if anno_dir is None:
		# weak fallback; user should pass explicitly in most cases
		guess = exp_dir.parent / 'annos'
		anno_dir = guess
	if not anno_dir.exists():
		print(f"[ERROR] 标注目录不存在：{anno_dir}（请用 --anno_dir 显式指定 data/.../annos）")
		return

	try:
		anno_files = sorted([p for p in anno_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
	except Exception:
		anno_files = []
	if not anno_files:
		print(f"[WARNING] 标注目录中未发现 .json：{anno_dir}（可能会导致几乎无样本被处理）")

	mmdet_base = Path(args.mmdet_base)
	if not mmdet_base.exists():
		print(f"[ERROR] mmdet_files 目录不存在：{mmdet_base}（请用 --mmdet_base 指定）")
		return

	weather_dirs = _discover_weather_dirs(exp_dir, skip_suffix=str(args.skip_suffix))
	print(f"[INFO] 发现天气结果文件夹：{[w for (w, _) in weather_dirs]}")
	if not weather_dirs:
		print(f"[ERROR] 未发现任何 final_*_images_* 目录于：{exp_dir}")
		return

	detectors_to_evaluate = list(DETECTOR_PATHS.keys()) if args.all_detectors else [args.detector]
	print(f"[INFO] 将评估的检测器：{detectors_to_evaluate}")

	method_name = exp_dir.name

	all_records: list[dict] = []
	all_preds_for_map: list[list[np.ndarray]] = []
	all_gts_for_map: list[dict] = []
	target_class_idx_global: int | None = None

	for det_name in detectors_to_evaluate:
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		try:
			detector = _init_one_detector(det_name, args.device, mmdet_base=mmdet_base)
		except Exception as e:
			print(f"[ERROR] 初始化检测器失败：{det_name}，原因：{e}")
			continue

		for weather_name, wdir in weather_dirs:
			recs, preds, gts, tc_idx = _evaluate_weather_dir_collect(
				detector=detector,
				detector_name=det_name,
				method_name=method_name,
				weather_dir=wdir,
				weather_name=weather_name,
				anno_dir=anno_dir,
				anno_files=anno_files,
				target_class_name=args.target_class_name,
				score_thresh=args.score_thresh,
			)

			if target_class_idx_global is None:
				target_class_idx_global = tc_idx
			elif target_class_idx_global != tc_idx:
				print("[WARNING] 目标类别 index 不一致（理论上不应发生），将继续使用首次的 index")

			all_records.extend(recs)
			all_preds_for_map.extend(preds)
			all_gts_for_map.extend(gts)

		# release detector
		try:
			del detector
		except Exception:
			pass

	if not all_records or target_class_idx_global is None:
		print("[ERROR] 没有获得任何可用结果（可能是标注匹配失败或目录为空）")
		return

	output_file = Path(args.output_file) if str(args.output_file).strip() else (exp_dir / 'evaluation_results_rpga.txt')
	_write_txt_report(
		output_file=output_file,
		records=all_records,
		preds_for_map=all_preds_for_map,
		gts_for_map=all_gts_for_map,
		target_class_idx=target_class_idx_global,
	)


if __name__ == '__main__':
	main()

