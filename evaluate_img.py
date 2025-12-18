from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import ReadError

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

# =================================================================================
# Helper Functions
# =================================================================================

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a folder of images with an MMDet detector')
    parser.add_argument('--image_dir', default='./EXP_EVAL/rauca', help='Path to the directory containing images to evaluate.')
    parser.add_argument('--anno_dir', default='./EXP_EVAL/annos', help='Path to the directory containing corresponding LabelMe annotations (.json).')
    parser.add_argument('--detector', type=str, default='detr', choices=list(DETECTOR_PATHS.keys()), help='MMDetection model to use.')
    parser.add_argument('--all_detectors', default=True, action='store_true', help='Evaluate on all available detectors. If set, --detector is ignored.')
    parser.add_argument('--target_class_name', type=str, default='car', help='Target class name for ASR calculation.')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold for considering a detection valid.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--output_file', default='evaluation_results.json', help='File to save the evaluation results.')
    return parser.parse_args()

def evaluate_detector(detector_name, image_dir, anno_dir, target_class_name='car', score_thresh=0.5, device='cuda:0'):
    """
    Evaluates a single detector on the given dataset.
    Returns a dictionary containing metrics.
    """
    print(f"[INFO] Initializing '{detector_name}' detector...")
    base_path = Path('./mmdet_files')
    
    selected_detector = DETECTOR_PATHS.get(detector_name)
    if selected_detector is None:
        print(f"[ERROR] Detector '{detector_name}' not supported.")
        return None

    cfg_path = str(base_path / selected_detector['config'])
    ckpt_path = str(base_path / selected_detector['ckpt'])

    try:
        detector = init_detector(cfg_path, ckpt_path, device=device)
        if not hasattr(detector, 'CLASSES'):
            detector.CLASSES = coco_classes
        print("[INFO] Detector initialized successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Failed to find detector files for {detector_name}. Make sure '{base_path}' is accessible.")
        print(f"[ERROR] Details: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector {detector_name}: {e}")
        return None

    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)
    
    if not image_dir.exists():
         print(f"[ERROR] Image directory '{image_dir}' does not exist.")
         return None

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    if not image_paths:
        print(f"[ERROR] No images found in '{image_dir}'.")
        return None

    try:
        target_class_idx = coco_classes.index(target_class_name)
    except ValueError:
        print(f"[ERROR] Target class '{target_class_name}' not found in COCO classes.")
        return None

    all_preds_for_map = []
    all_gts_for_map = []
    successful_attacks = 0
    images_processed = 0

    print(f"[INFO] Found {len(image_paths)} images. Starting evaluation for {detector_name}...")

    for img_path in tqdm(image_paths, ncols=100, desc=f"Eval {detector_name}"):
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

        all_gts_for_map.append({
            'bboxes': gt_bboxes,
            'labels': np.array([gt_label_idx] * len(gt_bboxes))
        })

        # Run inference
        try:
            img_np = np.array(Image.open(img_path).convert('RGB'))
            result = inference_detector(detector, img_np)
            pred_instances = result.pred_instances
        except Exception as e:
             print(f"\n[WARNING] Error processing {img_path.name}: {e}")
             continue

        # ASR calculation (updated):
        # 攻击成功当且仅当：不存在“分数≥阈值 且 与任一GT IoU≥0.5”的目标类预测
        is_attack_successful = True
        score_mask = pred_instances.scores >= score_thresh
        class_mask = pred_instances.labels == target_class_idx
        match_indices = (score_mask & class_mask).nonzero(as_tuple=False).squeeze(1)
        if match_indices.numel() > 0:
            pred_bboxes_tc = pred_instances.bboxes[match_indices]  # [K,4]
            if gt_label_idx == target_class_idx and gt_bboxes is not None and len(gt_bboxes) > 0:
                # 将 GT 转为与预测相同设备/类型的 Tensor
                gt_t = torch.from_numpy(gt_bboxes).to(pred_bboxes_tc.device, dtype=pred_bboxes_tc.dtype)
                # 计算每个预测框与任一 GT 的最大 IoU
                max_iou = torch.zeros(pred_bboxes_tc.shape[0], device=pred_bboxes_tc.device)
                for g in gt_t:
                    ious = compute_iou(pred_bboxes_tc, g.unsqueeze(0))  # [K]
                    max_iou = torch.maximum(max_iou, ious)
                if (max_iou >= 0.5).any():
                    is_attack_successful = False
        
        if is_attack_successful:
            successful_attacks += 1
            
        # Prepare predictions for mAP calculation
        num_classes = len(detector.CLASSES)
        pred_for_map = [np.empty((0, 5), dtype=np.float32) for _ in range(num_classes)]
        for i in range(num_classes):
            class_indices = (pred_instances.labels == i)
            if class_indices.any():
                boxes = pred_instances.bboxes[class_indices].cpu().numpy()
                scores = pred_instances.scores[class_indices].cpu().numpy()
                pred_for_map[i] = np.hstack([boxes, scores[:, np.newaxis]])
        all_preds_for_map.append(pred_for_map)
        images_processed += 1

    if images_processed == 0:
        print(f"\n[WARNING] No images were processed for {detector_name}. Check if annotations match image filenames.")
        return None

    # ASR
    asr = successful_attacks / images_processed
    
    # mAP @ 0.5
    eval_results = calculate_ap_for_target_class(all_preds_for_map, all_gts_for_map, target_class_idx, iou_thr=0.5)
    ap50 = eval_results.get('AP50', float('nan'))
    
    metrics = {
        'detector': detector_name,
        'target_class': target_class_name,
        'ASR': asr,
        'AP50': ap50,
        'processed_images': images_processed,
        'successful_attacks': successful_attacks
    }
    
    print(f"\n--- Results for {detector_name} ---")
    print(f"Attack Success Rate (ASR) on '{target_class_name}': {asr:.4f} ({successful_attacks}/{images_processed})")
    print(f"Mean Average Precision (AP@0.5): {ap50:.4f}")
    print("-----------------------------------\n")
    
    return metrics

def main():
    args = parse_args()
    
    detectors_to_evaluate = []
    if args.all_detectors:
        detectors_to_evaluate = list(DETECTOR_PATHS.keys())
    else:
        detectors_to_evaluate = [args.detector]
        
    results = []
    
    print(f"[INFO] Detectors to evaluate: {detectors_to_evaluate}")
    
    for det_name in detectors_to_evaluate:
        # Free up memory if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        metric = evaluate_detector(
            det_name, 
            args.image_dir, 
            args.anno_dir, 
            args.target_class_name, 
            args.score_thresh, 
            args.device
        )
        if metric:
            results.append(metric)
            
    # Save results
    if results:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"[INFO] All results saved to {args.output_file}")
        except Exception as e:
             print(f"[ERROR] Failed to save results: {e}")
    else:
        print("[WARNING] No results obtained.")

if __name__ == '__main__':
    main()
