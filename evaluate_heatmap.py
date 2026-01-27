# -*- coding: utf-8 -*-
"""
本脚本用于评估在多天气（Multi-Weather）条件下，应用了对抗性纹理的3D高斯模型的攻击效果，
并按 (angle, pitch) 组合生成热力图。

功能概述:
- 批量评估：支持同时评估多个根目录下的图像，每个根目录包含不同的天气子文件夹（如'Dusk', 'Night', 'Rain'等）。
- 多检测器支持：可针对一系列预定义的目标检测器（YOLOX, DETR, Faster R-CNN等）进行评估。
- 数据记录：记录每种方法在不同 (angle, pitch) 组合下的 AP@0.5（平均 distance，平均 weather）。
- 热力图生成：为每个方法生成一张热力图，以 (angle, pitch) 为横纵坐标轴的 AP@0.5。
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm
from utils.main_utils import calculate_ap_for_target_class, coco_classes, load_labelme_annotation, compute_iou

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    parser = argparse.ArgumentParser(description='评估 *mw2 数据，按 (angle, pitch) 组合生成热力图')
    parser.add_argument('--exp_eval_dir', default='./EXP_EVAL', help='EXP_EVAL 根目录路径（包含 *_mw2 方法文件夹与 annos 标注文件夹）')
    parser.add_argument('--anno_dir', default='./EXP_EVAL/annos', help='包含 LabelMe 标注 (.json) 的目录路径')
    parser.add_argument('--detector', type=str, default='yolox', choices=list(DETECTOR_PATHS.keys()), help='单个检测器（当不启用 --all_detectors 时使用）')
    parser.add_argument('--all_detectors', action=argparse.BooleanOptionalAction, default=True, help='是否评估所有检测器（默认开启）。若关闭则仅评估 --detector（可用 --no-all_detectors）')
    parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for ASR calculation.')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for considering a detection valid.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--output_file', default='evaluation_results_heatmap.txt', help='输出 txt 文件名/路径')
    parser.add_argument('--heatmap_dir', default='./heatmap_out', help='热力图输出目录')
    parser.add_argument('--mw2_suffix', default='mw2', help='只评估以该后缀结尾的方法文件夹（默认 mw2，如 dta_mw2/ori_mw2 等）')
    parser.add_argument('--skip_prefix', default='_', help='跳过以该前缀开头的子目录（默认 _，用于忽略 _EnvironmentMaps 等）')
    parser.add_argument('--angle_step', type=int, default=20, help='方位角步长（度），用于热力图网格划分')
    parser.add_argument('--pitch_step', type=int, default=10, help='俯仰角步长（度），用于热力图网格划分')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='热力图图片格式')
    parser.add_argument('--dpi', type=int, default=500, help='DPI（仅PNG）')
    parser.add_argument('--fontsize', type=int, default=30, help='统一字体大小（用于坐标轴标签、刻度和colorbar）')
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
) -> tuple[list[dict], int]:
    """
    返回：
      - records：逐图元信息（含 angle, pitch, distance, weather, preds, gts）
      - target_class_idx：目标类别 index（全局一致，便于后面聚合）
    """
    try:
        target_class_idx = coco_classes.index(target_class_name)
    except ValueError:
        raise ValueError(f"目标类别不在 COCO classes 中：{target_class_name}")

    image_paths = sorted([p for p in weather_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES])
    if not image_paths:
        return [], target_class_idx

    local_records: list[dict] = []
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

        pitch, angle, distance = _parse_pitch_angle_distance(img_path.stem)

        # 收集预测和GT用于后续按 (angle, pitch) 组合计算 AP@0.5
        pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
        for i in range(num_classes):
            class_indices = (pred_instances.labels == i)
            if class_indices.any():
                boxes = pred_instances.bboxes[class_indices].cpu().numpy()
                scores = pred_instances.scores[class_indices].cpu().numpy()
                pred_for_map[i] = np.hstack([boxes, scores[:, np.newaxis]])

        gt_for_map = {
            'bboxes': gt_bboxes,
            'labels': np.array([gt_label_idx] * len(gt_bboxes))
        }

        local_records.append({
            'method': method_name,
            'detector': detector_name,
            'weather': weather_name,
            'image_name': img_path.name,
            'pitch': pitch,
            'angle': angle,
            'distance': distance,
            'pred_for_map': pred_for_map,
            'gt_for_map': gt_for_map,
        })

    return local_records, target_class_idx


def _aggregate_by_angle_pitch(
    records: list[dict],
    target_class_idx: int,
) -> dict[tuple[str, int | None, int | None], float]:
    """
    按 (method, angle, pitch) 组合聚合数据，计算整体 AP@0.5
    跨所有检测器、distance 和 weather 合并数据
    返回: {(method, angle, pitch): ap50_value}
    """
    def _make_dict():
        return {'preds': [], 'gts': []}
    
    aggregated = defaultdict(_make_dict)
    
    for r in records:
        key = (r['method'], r.get('angle'), r.get('pitch'))
        if r.get('angle') is None or r.get('pitch') is None:
            continue
        aggregated[key]['preds'].append(r['pred_for_map'])
        aggregated[key]['gts'].append(r['gt_for_map'])
    
    # 计算每个组合的整体 AP@0.5
    result = {}
    for key, data in aggregated.items():
        if not data['preds'] or not data['gts']:
            continue
        ap50 = calculate_ap_for_target_class(
            data['preds'], data['gts'], target_class_idx, iou_thr=0.5
        ).get('AP50', float('nan'))
        result[key] = ap50
    
    return result


def _write_txt_report(
    output_file: Path,
    aggregated_data: dict[tuple[str, int | None, int | None], float],
):
    """
    写入 txt 报告，记录每种方法在不同 (angle, pitch) 组合下的 AP@0.5
    """
    lines: list[str] = []
    lines.append("========== 热力图评估汇总（按 angle, pitch 组合）==========")
    
    # 按方法分组
    methods = sorted({k[0] for k in aggregated_data.keys()})
    lines.append(f"方法数量：{len(methods)}")
    lines.append("")
    
    for method in methods:
        lines.append(f"【方法：{method}】")
        
        # 获取该方法的所有 (angle, pitch) 组合
        method_keys = [(k[1], k[2]) for k in aggregated_data.keys() if k[0] == method]
        method_keys = sorted(set(method_keys), key=lambda x: (x[1] if x[1] is not None else -999, x[0] if x[0] is not None else -999))
        
        for angle, pitch in method_keys:
            key = (method, angle, pitch)
            ap50 = aggregated_data.get(key)
            if ap50 is None or np.isnan(ap50):
                continue
            
            angle_str = f"angle{angle}" if angle is not None else "angle?"
            pitch_str = f"pitch{pitch}" if pitch is not None else "pitch?"
            lines.append(
                f"组合 ({angle_str}, {pitch_str})：AP@0.5 = {ap50:.4f}"
            )
        
        lines.append("")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding='utf-8')
    print(f"[INFO] 已写出汇总到：{output_file}")


def _interpolate_missing_values(heatmap: np.ndarray) -> np.ndarray:
    """
    对热力图中的NaN值进行插值填充
    使用简单的最近邻插值
    """
    filled = heatmap.copy()
    mask = ~np.isnan(heatmap)
    
    if not mask.any():
        return filled
    
    # 找到所有NaN位置
    nan_indices = np.where(np.isnan(filled))
    
    if len(nan_indices[0]) == 0:
        return filled
    
    # 对于每个NaN位置，找到最近的有限值
    for i, j in zip(nan_indices[0], nan_indices[1]):
        found = False
        for radius in range(1, max(filled.shape)):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    if (0 <= ni < filled.shape[0] and 
                        0 <= nj < filled.shape[1] and 
                        not np.isnan(filled[ni, nj])):
                        filled[i, j] = filled[ni, nj]
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        # 如果还是没找到，使用全局平均值
        if not found:
            global_mean = np.nanmean(filled)
            if not np.isnan(global_mean):
                filled[i, j] = global_mean
    
    return filled


def _compute_heatmap_data(
    method_data: dict[tuple[int | None, int | None], float],
    angle_range: tuple[int, int] = (0, 360),
    pitch_range: tuple[int, int] = (0, 90),
    angle_step: int = 20,
    pitch_step: int = 10,
    interpolate: bool = True,
) -> np.ndarray:
    """
    计算热力图数据矩阵
    
    Args:
        method_data: {(angle, pitch): ap50_value} 字典
        angle_range: (min_angle, max_angle)
        pitch_range: (min_pitch, max_pitch)
        angle_step: 方位角步长
        pitch_step: 俯仰角步长
        interpolate: 是否对缺失值进行插值
    
    Returns:
        heatmap_matrix: shape (pitch_bins, angle_bins) 的numpy数组
    """
    min_angle, max_angle = angle_range
    min_pitch, max_pitch = pitch_range
    
    # 创建角度和俯仰角的bins
    angle_bins = np.arange(min_angle, max_angle + angle_step, angle_step)
    pitch_bins = np.arange(min_pitch, max_pitch + pitch_step, pitch_step)
    
    # 初始化热力图矩阵
    heatmap = np.full((len(pitch_bins) - 1, len(angle_bins) - 1), np.nan)
    
    # 填充数据
    for (angle, pitch), ap50_value in method_data.items():
        if angle is None or pitch is None or np.isnan(ap50_value):
            continue
        
        # 找到对应的bin索引
        angle_idx = np.digitize(angle, angle_bins) - 1
        pitch_idx = np.digitize(pitch, pitch_bins) - 1
        
        # 确保索引在有效范围内
        if 0 <= angle_idx < len(angle_bins) - 1 and 0 <= pitch_idx < len(pitch_bins) - 1:
            # 如果该位置已有数据，取平均（理论上不应该发生，因为每个组合应该只有一个值）
            if np.isnan(heatmap[pitch_idx, angle_idx]):
                heatmap[pitch_idx, angle_idx] = ap50_value
            else:
                heatmap[pitch_idx, angle_idx] = (heatmap[pitch_idx, angle_idx] + ap50_value) / 2
    
    # 对缺失值进行插值
    if interpolate:
        nan_count = np.isnan(heatmap).sum()
        if nan_count > 0:
            print(f"[INFO] 插值填充 {nan_count} 个缺失值")
            heatmap = _interpolate_missing_values(heatmap)
    
    return heatmap


def _plot_heatmap(
    heatmap: np.ndarray,
    angle_bins: np.ndarray,
    pitch_bins: np.ndarray,
    method_name: str,
    output_path: Path,
    format: str = "png",
    dpi: int = 500,
    fontsize: int = 30,
):
    """
    绘制热力图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建自定义colormap：蓝色（容易）-> 红色（难）
    colors = ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe',  # 蓝色系
              '#fef3c7', '#fde68a', '#f59e0b', '#ef4444', '#dc2626', '#991b1b']  # 黄色到红色
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # 绘制热力图
    im = ax.imshow(
        heatmap,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='bilinear',
        extent=[angle_bins[0], angle_bins[-1], pitch_bins[0], pitch_bins[-1]],
    )
    
    # 设置标签（只保留英文，去掉中文和单位）
    ax.set_xlabel('Azimuth', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Pitch', fontsize=fontsize, fontweight='bold')
    ax.set_title(f'Spherical Unwrapping Heatmap - {method_name}\n(AP@0.5: Red=Hard, Blue=Easy)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 设置刻度
    ax.set_xticks(np.arange(0, 361, 40))
    ax.set_yticks(np.arange(0, 91, 10))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    
    # 统一设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Loss(Average)', fontsize=fontsize, fontweight='bold')
    # 设置colorbar刻度字体大小
    cbar.ax.tick_params(labelsize=fontsize)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 已保存热力图到：{output_path}")


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
    heatmap_dir = Path(args.heatmap_dir)

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
                recs, tc_idx = _evaluate_weather_dir_collect(
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

    if not all_records or target_class_idx_global is None:
        print("[ERROR] 没有获得任何可用结果（可能是标注匹配失败或目录为空）")
        return

    # 按 (method, angle, pitch) 组合聚合数据，计算整体 AP@0.5
    aggregated_data = _aggregate_by_angle_pitch(all_records, target_class_idx_global)
    
    if not aggregated_data:
        print("[ERROR] 聚合后没有可用数据")
        return
    
    # 写入 txt 报告
    _write_txt_report(output_file, aggregated_data)
    
    # 为每个方法生成热力图
    methods = sorted({k[0] for k in aggregated_data.keys()})
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    for method in methods:
        print(f"[INFO] 为方法 {method} 生成热力图...")
        
        # 提取该方法的数据
        method_data = {}
        for (m, angle, pitch), ap50_value in aggregated_data.items():
            if m == method:
                method_data[(angle, pitch)] = ap50_value
        
        if not method_data:
            print(f"[WARNING] 方法 {method} 没有数据，跳过")
            continue
        
        # 计算热力图矩阵
        heatmap = _compute_heatmap_data(
            method_data,
            angle_range=(0, 360),
            pitch_range=(0, 90),
            angle_step=args.angle_step,
            pitch_step=args.pitch_step,
            interpolate=True,
        )
        
        # 生成bins
        angle_bins = np.arange(0, 361, args.angle_step)
        pitch_bins = np.arange(0, 91, args.pitch_step)
        
        # 绘制并保存
        safe_method_name = method.replace('/', '_').replace('\\', '_')
        output_path = heatmap_dir / f"heatmap_{safe_method_name}.{args.format}"
        _plot_heatmap(
            heatmap,
            angle_bins,
            pitch_bins,
            method,
            output_path,
            format=args.format,
            dpi=args.dpi,
            fontsize=args.fontsize,
        )
    
    print(f"[DONE] 所有热力图已保存到：{heatmap_dir}")

if __name__ == '__main__':
    main()

