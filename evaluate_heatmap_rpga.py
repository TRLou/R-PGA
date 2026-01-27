# -*- coding: utf-8 -*-
"""
评估 RPGA/RGA 输出目录下的"最终多天气渲染结果"，并按 (angle, pitch) 组合生成热力图。

与 `evaluate_heatmap.py` 的主要区别：
- 输入不是 EXP_EVAL 下的 *_mw2 方法文件夹
- 而是给定一个运行输出目录，例如：RGA_output/1218_191840_Beijing
- 脚本会自动发现其下的多天气结果文件夹，例如：
    final_full_images_Dark/
    final_full_images_Foggy/
    ...
- 方法名固定为 "R-PGA"
- 生成热力图保存到 heatmap_out 目录
"""

from __future__ import annotations

import argparse
import glob
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

from utils.main_utils import (
    calculate_ap_for_target_class,
    coco_classes,
    load_labelme_annotation,
    compute_iou,
)

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

METHOD_NAME = 'R-PGA'  # 固定方法名

# =================================================================================
# Helper Functions
# =================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 RGA_output/<timestamp> 目录下的多天气结果，生成热力图。支持使用通配符对多个目录进行批量评估。"
    )
    parser.add_argument(
        '--exp_dir',
        default='./RGA_output/paper_exp',
        help='单个运行输出目录，或使用通配符 (e.g., "RGA_output/1224_*") 的模式，用于批量处理。'
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
        default='yolox',
        choices=list(DETECTOR_PATHS.keys()),
        help='单个检测器（当不启用 --all_detectors 时使用）'
    )
    parser.add_argument(
        '--all_detectors',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='是否评估所有检测器（默认关闭）。若开启则评估所有检测器（可用 --all_detectors）'
    )
    parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for ASR calculation.')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for considering a detection valid.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument(
        '--heatmap_dir',
        default='./heatmap_out',
        help='热力图输出目录'
    )
    parser.add_argument('--skip_suffix', default='_vis', help='跳过以该后缀结尾的结果文件夹（默认 _vis，用于忽略可视化目录）。设置为空则不跳过。')
    parser.add_argument('--angle_step', type=int, default=20, help='方位角步长（度），用于热力图网格划分')
    parser.add_argument('--pitch_step', type=int, default=10, help='俯仰角步长（度），用于热力图网格划分')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='热力图图片格式')
    parser.add_argument('--dpi', type=int, default=500, help='DPI（仅PNG）')
    parser.add_argument('--fontsize', type=int, default=30, help='统一字体大小（用于坐标轴标签、刻度和colorbar）')
    return parser.parse_args()


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
    与 evaluate_img_mw2.py 保持一致的"宽松匹配"：
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

    for img_path in tqdm(image_paths, ncols=100, desc=f"{METHOD_NAME}/{weather_name}/{detector_name}"):
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
            'method': METHOD_NAME,
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


def run_evaluation_on_directory(exp_dir: Path, args: argparse.Namespace, is_batch_mode: bool):
    """对单个实验目录执行评估并生成热力图"""
    if not exp_dir.is_dir():
        print(f"[ERROR] exp_dir 不是一个有效目录：{exp_dir}")
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

    all_records: list[dict] = []
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
            recs, tc_idx = _evaluate_weather_dir_collect(
                detector=detector,
                detector_name=det_name,
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

        # release detector
        try:
            del detector
        except Exception:
            pass

    if not all_records or target_class_idx_global is None:
        print("[ERROR] 没有获得任何可用结果（可能是标注匹配失败或目录为空）")
        return

    # 按 (method, angle, pitch) 组合聚合数据，计算整体 AP@0.5
    aggregated_data = _aggregate_by_angle_pitch(all_records, target_class_idx_global)
    
    if not aggregated_data:
        print("[ERROR] 聚合后没有可用数据")
        return
    
    # 为方法生成热力图
    heatmap_dir = Path(args.heatmap_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 为方法 {METHOD_NAME} 生成热力图...")
    
    # 提取该方法的数据
    method_data = {}
    for (m, angle, pitch), ap50_value in aggregated_data.items():
        if m == METHOD_NAME:
            method_data[(angle, pitch)] = ap50_value
    
    if not method_data:
        print(f"[WARNING] 方法 {METHOD_NAME} 没有数据，跳过")
        return
    
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
    safe_method_name = METHOD_NAME.replace('/', '_').replace('\\', '_')
    output_path = heatmap_dir / f"heatmap_{safe_method_name}.{args.format}"
    _plot_heatmap(
        heatmap,
        angle_bins,
        pitch_bins,
        METHOD_NAME,
        output_path,
        format=args.format,
        dpi=args.dpi,
        fontsize=args.fontsize,
    )
    
    print(f"[DONE] 热力图已保存到：{output_path}")


def main():
    args = parse_args()

    exp_dir_pattern = str(args.exp_dir)
    exp_dirs = [Path(p) for p in sorted(glob.glob(exp_dir_pattern)) if Path(p).is_dir()]

    if not exp_dirs:
        print(f"[ERROR] 未找到与模式 '{exp_dir_pattern}' 匹配的目录")
        return

    is_batch_mode = len(exp_dirs) > 1
    if is_batch_mode:
        print(f"[INFO] 发现 {len(exp_dirs)} 个匹配的实验目录，将进行批量处理...")

    for i, exp_dir in enumerate(exp_dirs):
        if is_batch_mode:
            print("\n" + "=" * 80)
            print(f"[BATCH {i + 1}/{len(exp_dirs)}] 正在处理目录: {exp_dir.name}")
            print("=" * 80)
        
        run_evaluation_on_directory(exp_dir, args, is_batch_mode)


if __name__ == '__main__':
    main()

