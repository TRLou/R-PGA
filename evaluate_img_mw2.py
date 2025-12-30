# -*- coding: utf-8 -*-
"""
本脚本用于评估在多天气（Multi-Weather）条件下，应用了对抗性纹理的3D高斯模型的攻击效果。

功能概述:
- 批量评估：支持同时评估多个根目录下的图像，每个根目录包含不同的天气子文件夹（如'Dusk', 'Night', 'Rain'等）。
- 多检测器支持：可针对一系列预定义的目标检测器（YOLOX, DETR, Faster R-CNN等）进行评估。
- 智能标注匹配：能够根据图像文件名（包含天气信息）与基础标注文件名进行灵活匹配，无需为每张多天气图像单独提供标注。
- 核心指标：计算并报告攻击成功率（ASR）和平均精度（AP@0.5），以量化攻击性能。
- 结果聚合：将所有评估结果（按根目录、天气、检测器分组）统一保存到一个JSON文件中，便于后续分析。
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
from utils.main_utils import calculate_ap_for_target_class, coco_classes, load_labelme_annotation, compute_iou

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

IMG_SUFFIXES = {'.png', '.jpg', '.jpeg'}

# =================================================================================
# Helper Functions
# =================================================================================

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='在 EXP_EVAL 下批量评估 *mw2 数据，并输出 pitch/distance/weather/detector 四种对比汇总表（txt）')
    parser.add_argument('--exp_eval_dir', default='./EXP_EVAL', help='EXP_EVAL 根目录路径（包含 *_mw2 方法文件夹与 annos 标注文件夹）')
    parser.add_argument('--anno_dir', default='./EXP_EVAL/annos', help='包含 LabelMe 标注 (.json) 的目录路径')
    parser.add_argument('--detector', type=str, default='detr', choices=list(DETECTOR_PATHS.keys()), help='单个检测器（当不启用 --all_detectors 时使用）')
    parser.add_argument('--all_detectors', action=argparse.BooleanOptionalAction, default=True, help='是否评估所有检测器（默认开启）。若关闭则仅评估 --detector（可用 --no-all_detectors）')
    parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for ASR calculation.')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for considering a detection valid.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--output_file', default='evaluation_results_mw2_5table.txt', help='输出 txt 文件名/路径（中文汇总，不输出 json）')
    parser.add_argument('--mw2_suffix', default='mw2', help='只评估以该后缀结尾的方法文件夹（默认 mw2，如 dta_mw2/ori_mw2 等）')
    parser.add_argument('--skip_prefix', default='_', help='跳过以该前缀开头的子目录（默认 _，用于忽略 _EnvironmentMaps 等）')
    return parser.parse_args()

def _parse_method_name(dir_name: str) -> str:
    if dir_name.endswith('_mw2'):
        return dir_name[:-4]
    if dir_name.endswith('mw2'):
        return dir_name[:-3].rstrip('_-.')
    return dir_name


def _safe_int_from_regex(stem: str, pattern: str) -> int | None:
    m = re.search(pattern, stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_pitch_distance(stem: str) -> tuple[int | None, int | None]:
    # 支持：pitch5 / Pitch5 / ..._pitch5_... 以及 distance20 / Distance20 等
    pitch = _safe_int_from_regex(stem, r'(?i)pitch(\d+)')
    distance = _safe_int_from_regex(stem, r'(?i)distance(\d+)')
    return pitch, distance


def _parse_pitch_angle_distance(stem: str) -> tuple[int | None, int | None, int | None]:
    """
    Parse discrete physical metadata from image filename stem.
    Expected patterns like: ori_pitch5_angle240_distance5_sunny
    Angle aliases supported: angle / azimuth / azi
    """
    pitch = _safe_int_from_regex(stem, r'(?i)pitch(-?\d+)')
    angle = _safe_int_from_regex(stem, r'(?i)(?:angle|azimuth|azi)(-?\d+)')
    distance = _safe_int_from_regex(stem, r'(?i)distance(-?\d+)')
    return pitch, angle, distance


def _discover_mw2_root_dirs(exp_eval_dir: Path, mw2_suffix: str) -> list[Path]:
    if not exp_eval_dir.exists():
        return []
    dirs = []
    for p in exp_eval_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == 'annos':
            continue
        if p.name.endswith(str(mw2_suffix)):
            dirs.append(p)
    return sorted(dirs)


def _init_one_detector(detector_name: str, device: str):
    print(f"[INFO] 初始化检测器：{detector_name} ...")
    base_path = Path('./mmdet_files')
    selected = DETECTOR_PATHS.get(detector_name)
    if selected is None:
        raise ValueError(f"不支持的检测器：{detector_name}")
    cfg_path = str(base_path / selected['config'])
    ckpt_path = str(base_path / selected['ckpt'])
    detector = init_detector(cfg_path, ckpt_path, device=device)
    if not hasattr(detector, 'CLASSES'):
        detector.CLASSES = coco_classes
    return detector


def _match_annotation_path(img_path: Path, anno_files: list[Path], anno_dir: Path, weather_name: str | None) -> Path | None:
    # 根据 weather 仅以前缀匹配标注：
    # 1) 优先使用文件名中 weather 出现前的部分作为前缀
    # 2) 否则回退为移除最后一个下划线段后的前缀
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
    """
    返回：
      - records：逐图元信息（含 attack_successful）
      - preds_for_map：逐图预测（mAP 计算用）
      - gts_for_map：逐图 GT（mAP 计算用）
      - target_class_idx：目标类别 index（全局一致，便于后面聚合）
    """
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

        # 强制只统计目标类别的样本（避免混入非 car 的标注导致 ASR/AP 语义混乱）
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

        pitch, angle, distance = _parse_pitch_angle_distance(img_path.stem)

        local_records.append({
            'method': method_name,
            'weather': weather_name,
            'detector': detector_name,
            'image_name': img_path.name,
            'pitch': pitch,
            'angle': angle,
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
    all_angles = sorted({r['angle'] for r in records if r.get('angle') is not None})
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
    lines.append("========== MW2 评估汇总（中文 txt）==========")
    lines.append(f"总样本数（逐图统计）：{len(records)}")
    lines.append(
        f"方法数量：{len(methods)}，天气数量：{len(all_weathers)}，距离种类：{len(all_distances)}，俯仰角种类：{len(all_pitches)}，方位角种类：{len(all_angles)}，检测器数量：{len(all_detectors)}"
    )
    lines.append("")

    # 1) pitch 对比表
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

    # 2) distance 对比表
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

    # 3) weather 对比表
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

    # 4) detector 对比表
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

    # 5) angle 对比表
    lines.append("========== 表 5：控制变量【方位角 angle】==========")
    for m in methods:
        lines.append(f"【方法：{m}】")
        for a in all_angles:
            idxs = [i for i in idxs_where(method=m) if records[i].get('angle') == a]
            asr, ap50, n, succ = _compute_metrics_for_indices(idxs, records, preds_for_map, gts_for_map, target_class_idx)
            if n == 0:
                continue
            lines.append(f"变量方位角：angle{a}，{m}方法的平均ASR是{asr:.4f}，平均AP@0.5是{ap50:.4f}（成功{succ}/{n}）")
        lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding='utf-8')
    print(f"[INFO] 已写出中文汇总到：{output_file}")

def main():
    args = parse_args()
    
    detectors_to_evaluate = []
    if args.all_detectors:
        detectors_to_evaluate = list(DETECTOR_PATHS.keys())
    else:
        detectors_to_evaluate = [args.detector]

    exp_eval_dir = Path(args.exp_eval_dir)
    anno_dir = Path(args.anno_dir)
    output_file = Path(args.output_file)

    root_image_dirs = _discover_mw2_root_dirs(exp_eval_dir, args.mw2_suffix)
    print(f"[INFO] 将评估的检测器：{detectors_to_evaluate}")
    print(f"[INFO] 发现 mw2 方法目录：{[p.name for p in root_image_dirs]}")
    if not root_image_dirs:
        print("[ERROR] 未发现任何以 mw2 结尾的方法目录，请检查 --exp_eval_dir / --mw2_suffix")
        return

    try:
        anno_files = sorted([p for p in anno_dir.iterdir() if p.is_file() and p.suffix.lower() == '.json'])
    except Exception:
        anno_files = []
    if not anno_files:
        print(f"[WARNING] 标注目录中未发现 .json：{anno_dir}（可能会导致几乎无样本被处理）")

    all_records: list[dict] = []
    all_preds_for_map: list[list[np.ndarray]] = []
    all_gts_for_map: list[dict] = []
    target_class_idx_global: int | None = None

    # 为避免显存爆炸：按 detector 逐个加载 & 跑完整数据集
    for det_name in detectors_to_evaluate:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            detector = _init_one_detector(det_name, args.device)
        except Exception as e:
            print(f"[ERROR] 初始化检测器失败：{det_name}，原因：{e}")
            continue

        for root_dir in root_image_dirs:
            method_name = _parse_method_name(root_dir.name)

            # 天气子目录：只取第一层目录，并跳过 _EnvironmentMaps 这类
            try:
                weather_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith(args.skip_prefix)])
            except Exception as e:
                print(f"[WARNING] 无法列出子目录：{root_dir}，原因：{e}")
                continue

            for weather_dir in weather_dirs:
                weather_name = weather_dir.name
                recs, preds, gts, tc_idx = _evaluate_weather_dir_collect(
                    detector=detector,
                    detector_name=det_name,
                    method_name=method_name,
                    weather_dir=weather_dir,
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

    if not all_records or target_class_idx_global is None:
        print("[ERROR] 没有获得任何可用结果（可能是标注匹配失败或目录为空）")
        return

    _write_txt_report(
        output_file=output_file,
        records=all_records,
        preds_for_map=all_preds_for_map,
        gts_for_map=all_gts_for_map,
        target_class_idx=target_class_idx_global,
    )

if __name__ == '__main__':
    main()
