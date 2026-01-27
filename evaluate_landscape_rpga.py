# -*- coding: utf-8 -*-
"""
评估 RPGA/RGA 输出目录下的"最终多天气渲染结果"，生成 loss landscape 图。

与 `evaluate_landscape_mw2.py` 的主要区别：
- 输入不是 EXP_EVAL 下的 *_mw2 方法文件夹
- 而是给定一个运行输出目录，例如：RGA_output/1218_191840_Beijing
- 脚本会自动发现其下的多天气结果文件夹，例如：
    final_full_images_Dark/
    final_full_images_Foggy/
    ...
- 方法名固定为 "R-PGA"
- 生成 landscape 图保存到 landscape_out 目录
- 横坐标为 (pitch, distance)，纵坐标为 (angle, weather)

功能模式：
- 计算模式（--skip_computation=False，默认）：
    * 计算所有图像的loss值
    * 将loss记录保存到 landscape_out/tables/<exp_dir_name>_<detector>.csv
    * 绘制landscape可视化图
  
- 可视化模式（--skip_computation=True）：
    * 从 landscape_out/tables/ 目录读取已保存的CSV文件
    * 直接绘制landscape可视化图（跳过计算步骤）
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter

from utils.main_utils import (
    coco_classes,
    load_labelme_annotation,
    calculate_ap_for_target_class,
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
METHOD_NAME = 'PGA'  # 固定方法名

# =================================================================================
# Helper Functions
# =================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 RGA_output/<timestamp> 目录下的多天气结果，生成 loss landscape 图。支持使用通配符对多个目录进行批量评估。"
    )
    parser.add_argument(
        '--exp_dir',
        default='./RGA_output/PGA',
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
        default='yolov3',
        choices=list(DETECTOR_PATHS.keys()),
        help='单个检测器（当不启用 --all_detectors 时使用）'
    )
    parser.add_argument(
        '--all_detectors',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否评估所有检测器（默认关闭）。若开启则评估所有检测器'
    )
    parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for loss calculation.')
    parser.add_argument('--score_thresh', type=float, default=0.01, help='Score threshold for considering a detection valid.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument(
        '--landscape_dir',
        default='./landscape_out',
        help='Landscape 图输出目录'
    )
    parser.add_argument('--skip_suffix', default='_vis', help='跳过以该后缀结尾的结果文件夹（默认 _vis，用于忽略可视化目录）。设置为空则不跳过。')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='Landscape 图片格式')
    parser.add_argument('--dpi', type=int, default=800, help='DPI（仅PNG，推荐800以上以获得最佳艺术化效果）')
    parser.add_argument('--fontsize', type=int, default=30, help='统一字体大小（用于坐标轴标签、刻度和colorbar）')
    parser.add_argument('--skip_computation', action=argparse.BooleanOptionalAction, default=False, help='如果开启，跳过AP@0.5计算，直接从 landscape_out/data/ 目录读取已保存的数据并绘制可视化。如果关闭，则计算AP@0.5并保存到数据文件，然后绘制可视化。')
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
        # Ignore special/internal folders if any
        if weather.startswith('_') or 'EnvironmentMaps' in weather:
            continue
        out.append((weather, p))
    return sorted(out, key=lambda x: x[0].lower())


def _match_annotation_path(img_path: Path, anno_files: list[Path], anno_dir: Path, weather_name: str | None) -> Path | None:
    """
    与 evaluate_img_mw2.py 保持一致的"宽松匹配"
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


def _save_aggregated_data(aggregated_data: dict[tuple[str, int | None, int | None], float], output_path: Path):
    """
    保存聚合后的数据到JSON文件
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为可序列化的格式
    data_dict = {}
    for (method, angle, pitch), ap50_value in aggregated_data.items():
        key = f"{method}_{angle}_{pitch}"
        data_dict[key] = {
            'method': method,
            'angle': angle,
            'pitch': pitch,
            'ap50': float(ap50_value) if not np.isnan(ap50_value) else None
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] 已保存聚合数据到：{output_path}")


def _load_aggregated_data(input_path: Path) -> dict[tuple[str, int | None, int | None], float]:
    """
    从JSON文件加载聚合后的数据
    """
    if not input_path.exists():
        return {}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    aggregated_data = {}
    for key, value in data_dict.items():
        method = value['method']
        angle = value['angle']
        pitch = value['pitch']
        ap50 = value['ap50']
        if ap50 is not None:
            aggregated_data[(method, angle, pitch)] = float(ap50)
    
    print(f"[INFO] 从 {input_path} 加载了 {len(aggregated_data)} 条数据")
    return aggregated_data


def _interpolate_missing_values(landscape: np.ndarray) -> np.ndarray:
    """对缺失值进行插值填充"""
    filled = landscape.copy()
    mask = ~np.isnan(landscape)
    
    if not mask.any():
        return filled
    
    nan_indices = np.where(np.isnan(filled))
    
    if len(nan_indices[0]) == 0:
        return filled
    
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
        
        if not found:
            global_mean = np.nanmean(filled)
            if not np.isnan(global_mean):
                filled[i, j] = global_mean
    
    return filled


def _compute_landscape_data(
    method_data: dict[tuple[int | None, int | None], float],
    angle_range: tuple[int, int] = (0, 360),
    pitch_range: tuple[int, int] = (0, 90),
    angle_step: int = 20,
    pitch_step: int = 10,
    smooth_interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 landscape 数据矩阵，并生成用于 3D 绘制的超光滑网格
    
    使用 RBF 插值确保完全覆盖整个平面，无空洞
    横坐标为 Azimuth (angle)，纵坐标为 Pitch
    """
    # 准备原始数据点
    points = []
    values = []
    
    for (angle, pitch), ap50_value in method_data.items():
        if angle is None or pitch is None or np.isnan(ap50_value):
            continue
        points.append([float(angle), float(pitch)])
        values.append(ap50_value)
    
    if not points:
        return None, None, None
    
    points = np.array(points)
    values = np.array(values)
    
    # 创建超密集网格以确保完全覆盖和光滑表面
    # 使用至少 200x200 的分辨率，或根据数据密度自动调整
    min_angle, max_angle = angle_range
    min_pitch, max_pitch = pitch_range
    
    base_res = max(200, int((max_angle - min_angle) / angle_step * 8), int((max_pitch - min_pitch) / pitch_step * 8))
    grid_resolution = min(base_res, 400)  # 限制最大分辨率以避免内存问题
    
    xi = np.linspace(min_angle, max_angle, grid_resolution)
    yi = np.linspace(min_pitch, max_pitch, grid_resolution)
    X, Y = np.meshgrid(xi, yi)
    
    # 首先尝试 RBF 插值（最光滑，最适合艺术化效果）
    if smooth_interpolate and len(points) >= 4:
        try:
            # 使用 thin-plate spline RBF（产生极其光滑的连续表面）
            # 这是最适合艺术化效果的插值方法，能产生有机的、流动的曲线
            rbf = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.0)
            grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
            Z = rbf(grid_points).reshape(X.shape)
            
            # 应用轻微的高斯平滑以消除任何剩余的不连续性，使表面像手工打磨的陶土
            # sigma 值较小以保持地形的尖锐特征（特别是高峰）
            Z = gaussian_filter(Z, sigma=0.6)
            
        except Exception as e:
            print(f"[INFO] RBF 插值失败，使用 cubic griddata: {e}")
            # 回退到 cubic 插值
            try:
                Z = griddata(points, values, (X, Y), method='cubic', fill_value=np.nanmean(values))
                nan_mask = np.isnan(Z)
                if nan_mask.any():
                    # 对 NaN 使用最近邻填充，然后再次 cubic 插值
                    Z_nn = griddata(points, values, (X, Y), method='nearest')
                    Z[nan_mask] = Z_nn[nan_mask]
                    # 第二次 cubic 插值使过渡更平滑
                    valid_mask = ~np.isnan(Z)
                    if valid_mask.sum() > 10:
                        valid_points = np.stack([X[valid_mask], Y[valid_mask]], axis=1)
                        valid_values = Z[valid_mask]
                        Z = griddata(valid_points, valid_values, (X, Y), method='cubic', fill_value=np.nanmean(values))
                # 应用高斯平滑
                Z = gaussian_filter(Z, sigma=1.0)
            except Exception:
                # 最终回退到 linear
                Z = griddata(points, values, (X, Y), method='linear', fill_value=np.nanmean(values))
                Z = gaussian_filter(Z, sigma=1.2)
    else:
        # 数据点太少，使用基础插值
        Z = griddata(points, values, (X, Y), method='linear', fill_value=np.nanmean(values))
        Z = gaussian_filter(Z, sigma=1.5)
    
    # 确保完全没有 NaN（完全覆盖整个平面）
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        # 使用最近邻填充剩余 NaN
        Z_nn = griddata(points, values, (X, Y), method='nearest')
        Z[nan_mask] = Z_nn[nan_mask]
    
    return X, Y, Z


def _plot_landscape(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    method_name: str,
    output_path: Path,
    format: str = "png",
    dpi: int = 500,
    fontsize: int = 30,
):
    """
    绘制高质量艺术化 3D AP@0.5 landscape 地形图
    
    特征：
    - 完全光滑的连续表面（无空洞）
    - 电影级光照效果
    - 艺术化渐变色映射
    - 横坐标为 Azimuth，纵坐标为 Pitch
    """
    # 使用更大的画布以获得更多细节，展现地形的精细起伏
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # 创建艺术化颜色映射：深靛蓝（谷底）-> 蓝绿 -> 米色/浅橙 -> 炽热红橙（高峰）
    # 使用更丰富的渐变以展现"尖锐的红橙色高峰"和"深蓝色的谷底"
    colors = [
        '#0a0019', '#1a0033', '#2d0066', '#400099',  # 深靛蓝到紫色（最深的谷底）
        '#4d00cc', '#0066ff', '#00aaff', '#00ddff',  # 深蓝到青色
        '#40ffe0', '#80ffcc', '#aaffcc', '#ccffcc',  # 浅蓝绿
        '#ffffcc', '#fff9aa', '#ffee88', '#ffe066',  # 米色到浅黄色
        '#ffcc44', '#ffaa22', '#ff8800', '#ff6600',  # 浅橙到橙色
        '#ff4400', '#ff2200', '#ff0000', '#ee0000',  # 红橙色
        '#cc0000', '#990000', '#660000'  # 深红色到暗红（最高峰）
    ]
    n_bins = 512  # 使用更高分辨率以获得极其平滑的渐变
    cmap = LinearSegmentedColormap.from_list('artistic_terrain', colors, N=n_bins)
    
    # 绘制极其光滑的表面（完全消除体素感，使用最密集网格）
    # shade=True 启用基于法向量的光照计算，matplotlib 会自动计算表面法向量
    # 并根据视角产生高光反射效果（类似陶瓷釉面的反射）
    # 通过调整视角（azim=320）可以从左上角方向观察，使受光面产生强烈高光
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        alpha=1.0,
        linewidth=0,
        antialiased=True,
        shade=True,  # 启用基于法向量的光照计算（产生高光反射效果）
        rstride=1,   # 使用最密集的网格，完全消除体素感
        cstride=1,
        edgecolor='none',  # 无边缘线，保证完全光滑
        vmin=0.0,  # 固定范围为0-1
        vmax=1.0,
        norm=None,
    )
    
    # 设置坐标轴标签（只保留英文，去掉中文和单位）
    ax.set_xlabel('Azimuth', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Pitch', fontsize=fontsize, fontweight='bold')
    ax.set_zlabel('Loss(Average)', fontsize=fontsize, fontweight='bold')
    
    # 移除所有刻度（只保留坐标轴名字）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # 移除所有背景平面和网格
    ax.grid(False)  # 完全关闭网格
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    
    # 移除坐标轴线条
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # 设置视角为俯视透视，优化角度以最大化光照效果
    # elev=60-70 俯视角度（角度越高越能展现地形起伏）
    # azim=315-330 从左上角方向看，配合光照产生强烈高光反射
    # 这个角度模拟从左上角打入的强烈定向光源，使受光面产生陶瓷釉面般的高光
    ax.view_init(elev=65, azim=320)
    
    # 添加 colorbar，固定范围为0-1
    cbar = plt.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Loss(Average)', fontsize=fontsize, fontweight='bold')
    # 设置colorbar刻度字体大小，并固定范围为0-1
    cbar.set_ticks([0.0, 1.0])
    cbar.ax.tick_params(labelsize=fontsize)
    
    # 设置纯白背景（无缝、纯净的白色）
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 调整边距以获得最佳构图（完全填充画布）
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存为超高质量艺术化图像
    # 使用高 DPI 和优化设置以获得最佳视觉效果
    plt.savefig(
        output_path,
        format=format,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0,
        transparent=False,
        metadata={'Title': f'3D AP@0.5 Landscape - {method_name}'},
    )
    plt.close()
    print(f"[INFO] 已保存高质量艺术化 3D landscape 图到：{output_path}")


def run_evaluation_on_directory(exp_dir: Path, args: argparse.Namespace, is_batch_mode: bool):
    """对单个实验目录执行评估并生成 landscape 图"""
    if not exp_dir.is_dir():
        print(f"[ERROR] exp_dir 不是一个有效目录：{exp_dir}")
        return

    landscape_dir = Path(args.landscape_dir)
    landscape_dir.mkdir(parents=True, exist_ok=True)
    data_dir = landscape_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建数据文件路径（基于exp_dir和检测器）
    detectors_to_evaluate = list(DETECTOR_PATHS.keys()) if args.all_detectors else [args.detector]
    data_file = data_dir / f"aggregated_data_{exp_dir.name}_{'_'.join(detectors_to_evaluate)}.json"
    
    if args.skip_computation:
        # 可视化模式：从数据文件读取
        print("[INFO] 可视化模式：正在从数据文件读取数据...")
        aggregated_data = _load_aggregated_data(data_file)
        if not aggregated_data:
            print(f"[ERROR] 无法从 {data_file} 读取数据，请先运行计算模式")
            return
    else:
        # 计算模式：计算AP@0.5并保存
        print("[INFO] 计算模式：正在计算AP@0.5并保存数据...")
        
        # Resolve anno_dir
        anno_dir = Path(args.anno_dir) if str(args.anno_dir).strip() else None
        if anno_dir is None:
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
        
        # 保存聚合数据
        _save_aggregated_data(aggregated_data, data_file)
    
    # 为方法生成 landscape 图
    print(f"[INFO] 为方法 {METHOD_NAME} 生成 landscape 图...")
    
    # 提取该方法的数据
    method_data = {}
    for (m, angle, pitch), ap50_value in aggregated_data.items():
        if m == METHOD_NAME:
            method_data[(angle, pitch)] = ap50_value
    
    if not method_data:
        print(f"[WARNING] 方法 {METHOD_NAME} 没有数据，跳过")
        return
    
    # 计算 landscape 3D 网格数据
    X, Y, Z = _compute_landscape_data(
        method_data,
        angle_range=(0, 360),
        pitch_range=(0, 90),
        angle_step=20,
        pitch_step=10,
        smooth_interpolate=True,
    )
    
    if X is None or Y is None or Z is None:
        print(f"[WARNING] 方法 {METHOD_NAME} 无法生成 3D 数据，跳过")
        return
    
    # 绘制并保存
    safe_method_name = METHOD_NAME.replace('/', '_').replace('\\', '_')
    output_path = landscape_dir / f"landscape_{safe_method_name}.{args.format}"
    _plot_landscape(
        X, Y, Z,
        METHOD_NAME,
        output_path,
        format=args.format,
        dpi=args.dpi,
        fontsize=args.fontsize,
    )
    
    print(f"[DONE] Landscape 图已保存到：{output_path}")


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

