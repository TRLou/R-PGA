import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix
)
import numpy as np
import torch.nn.functional as F
import math
from utils.sh_utils import RGB2SH
import cv2


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# coco_classes = {
#     1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
#     10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
#     18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
#     27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis",
#     36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard",
#     42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
#     50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
#     58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
#     67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
#     77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
#     85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
# }

def load_labelme_annotation(json_path: str):
    """Loads a single bounding box and label from a LabelMe JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    shapes = data.get('shapes', [])
    if not shapes:
        return None, None
    
    # Assuming one object per annotation file
    shape = shapes[0]
    label = shape.get('label')
    points = shape.get('points', [])
    
    if not label or not points or len(points) < 2:
        return None, None

    # LabelMe 可能是矩形（2点）或多边形（>2点），统一转为 bbox
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    
    return np.array([bbox], dtype=np.float32), label

main_colors = {
    'green': [
        (71,  76,  63),  # DG 0730
        (75,  80,  68),  # DG 0810
        (76,  88,  69),  # MG 0943
        (83,  91,  74),  # MG 1020
        (79,  95,  61),  # EG 1054
        (98, 106,  73),  # EG 1456
        (95,  98,  75),  # YG 1247
        (108, 111,  64), # YG 1560
    ],
    'brown': [
        (86,  80,  74),  # BE 0811
        (98,  80,  64),  # BE 0922
        (105, 92,  70),  # BE 1130
        (104, 93,  82),  # BE 1225
        (109, 93,  86),  # BE 1230
        (139, 109, 86),  # BE 1732
        (128, 114, 94),  # BE 1824
    ],
    'red_earth': [
        (117, 84,  62),  # RE 1125
        (133, 91,  74),  # RE 1328
    ],
    'yellow_earth': [
        (123, 110, 78),  # YE 1624
        (161, 124, 95),  # YE 2344
        (170, 135, 98),  # YE 2738
        (183, 145, 99),  # YE 3249
        (202, 150, 96),  # YE 3559
    ],
    'sand': [
        (145, 122, 87),  # SE 2139
        (145, 127, 104), # SE 2232
        (136, 133, 127), # SE 2425
        (157, 137, 107), # SE 2635
        (166, 154, 138), # SE 3340
        (181, 165, 141), # SE 3948
    ],
    'black': [
        (0, 0, 0), (5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20),
        (25, 25, 25), (30, 30, 30), (35, 35, 35), (40, 40, 40), (45, 45, 45),
        (50, 50, 50), (55, 55, 55)
    ],
    # 'neutral': [
    #     (69,  68,  68),  # BN 0606
    #     (235, 235, 233), # WN 8384
    # ],
}

# main_colors = {
#     'dark_green': [
#         (35, 45, 35), (40, 50, 40), (45, 55, 45), (50, 60, 50), (55, 65, 55),
#         (60, 70, 60), (65, 75, 65), (70, 80, 70), (75, 85, 75), (80, 90, 80),
#         (85, 95, 85), (90, 100, 90), (95, 105, 95)
#     ],
#     'black': [
#         (0, 0, 0), (5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20),
#         (25, 25, 25), (30, 30, 30), (35, 35, 35), (40, 40, 40), (45, 45, 45),
#         (50, 50, 50), (55, 55, 55)
#     ],
#     'coffee': [
#         (80, 52, 27), (85, 55, 30), (90, 58, 33), (95, 61, 36), (100, 64, 39),
#         (105, 67, 42), (110, 70, 45), (115, 73, 48), (120, 76, 51), (125, 79, 54),
#         (130, 82, 57), (135, 85, 60)
#     ],
#     'desert': [
#         (194, 178, 128), (200, 182, 132), (206, 186, 136), (212, 190, 140), (218, 194, 144),
#         (224, 198, 148), (230, 202, 152), (236, 206, 156), (242, 210, 160), (248, 214, 164),
#         (254, 218, 168), (220, 200, 145), (230, 210, 155)
#     ]
# }



main_colors_sh = {}
for name, rgb_list in main_colors.items():
    # 转成 Tensor 并归一化到 [0,1]
    rgb_tensor = torch.tensor(rgb_list, dtype=torch.float32) / 255.0  # shape [N,3]
    # 调用 RGB2SH，得到 shape [N, K, 3] 的 SH 系数
    sh_coeffs = RGB2SH(rgb_tensor)
    # 存入新的字典（可以直接存 Tensor，也可转成 list）
    main_colors_sh[name] = sh_coeffs  # 或者: sh_coeffs.tolist()

def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
    """
    批量 SLERP：
      q0, q1: (..., 4) 单位四元数
      t:     (..., 1) 插值权重 0→1
    返回 (...,4) 的插值四元数
    """
    # 计算点积并修正反向
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)            # (...,1)
    neg_mask = dot < 0
    q1 = torch.where(neg_mask, -q1, q1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # 角度与 sin
    theta0 = torch.acos(dot)                                  # (...,1)
    sin0   = torch.sin(theta0)                                # (...,1)

    # 当角度很小时，退化为线性插值
    small = sin0.abs() < eps
    # 正常 SLERP 部分
    theta = theta0 * t                                        # (...,1)
    s0 = torch.sin(theta0 - theta) / (sin0 + eps)             # (...,1)
    s1 = torch.sin(theta)      / (sin0 + eps)                 # (...,1)
    qs = s0 * q0 + s1 * q1                                    # (...,4)

    # 线性退化部分
    qs_lin = q0 + t * (q1 - q0)
    qs = torch.where(small, qs_lin, qs)

    # 归一化
    return qs / qs.norm(dim=-1, keepdim=True)

def random_submask(orig_mask, min_ratio: float = 0.2, max_ratio: float = 0.8):
    """
    从原始二值 mask 中随机提取一个子 mask，保持输入 shape [1,3,H,W]。

    参数：
      orig_mask   torch.Tensor 或 np.ndarray，shape [1,3,H,W]，值可为 0/1 或 0–255
      min_ratio   子区域相对原包围盒最小比例
      max_ratio   子区域相对原包围盒最大比例

    返回：
      new_mask    同类型同 shape 的子 mask
    """
    is_tensor = isinstance(orig_mask, torch.Tensor)
    # 转为 numpy 处理
    if is_tensor:
        device = orig_mask.device
        orig_np = orig_mask.detach().cpu().numpy()  # [1,3,H,W]
    else:
        orig_np = orig_mask

    # collapse batch dim then any over channels: [H,W]
    bin_mask = (orig_np > 0).any(axis=1)[0].astype(np.uint8)

    # 找到前景包围盒
    ys, xs = np.where(bin_mask)
    if len(xs) == 0:
        sub = np.zeros_like(bin_mask)
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bw, bh = x_max - x_min + 1, y_max - y_min + 1

        # 随机尺寸
        ratio = np.random.uniform(min_ratio, max_ratio)
        w = max(1, int(bw * ratio))
        h = max(1, int(bh * ratio))

        # 随机位置
        x0 = np.random.randint(x_min, x_max - w + 1) if x_max - w >= x_min else x_min
        y0 = np.random.randint(y_min, y_max - h + 1) if y_max - h >= y_min else y_min

        sub_box = bin_mask[y0:y0+h, x0:x0+w]

        # 随机选择形状
        if np.random.rand() < 0.5:
            new_sub = sub_box
        else:
            ellipse = np.zeros((h, w), dtype=np.uint8)
            center = (w//2, h//2)
            axes   = (max(1, w//2 - 1), max(1, h//2 - 1))
            cv2.ellipse(ellipse, center, axes, 0, 0, 360, 1, -1)
            new_sub = sub_box * ellipse

        sub = np.zeros_like(bin_mask)
        sub[y0:y0+h, x0:x0+w] = new_sub

    # 扩展回 [1,3,H,W]
    sub3 = np.stack([sub, sub, sub], axis=0)  # [3,H,W]
    new_mask_np = sub3[np.newaxis, ...]       # [1,3,H,W]

    if is_tensor:
        return torch.from_numpy(new_mask_np).to(device)
    else:
        return new_mask_np


def interpolate_trajectory(c2ws: torch.Tensor,
                           num_new: int) -> (torch.Tensor, torch.Tensor):
    """
    输入:
      c2ws    (N,3,4) 原始 c2w 矩阵
      num_new    int  期望插值帧数
    输出:
      c2ws_interp (num_new,3,4) 插值后的 c2w
      centers_interp (num_new,3) 插值后的相机中心
    """
    device = c2ws.device
    N = c2ws.shape[0]

    # 提取原始平移向量 (camera centers)
    centers = c2ws[..., :3, 3]    # (N,3)

    # 构造插值位置
    t_new = torch.linspace(0, N - 1, num_new, device=device)   # (num_new,)
    idx0  = t_new.floor().long()                               # (num_new,)
    idx1  = torch.clamp(idx0 + 1, max=N-1)
    alpha = (t_new - idx0.float()).unsqueeze(1)                # (num_new,1)

    # —— 平移插值 ——
    c0 = centers[idx0]      # (num_new,3)
    c1 = centers[idx1]
    centers_interp = (1 - alpha) * c0 + alpha * c1             # (num_new,3)

    # —— 旋转插值 ——
    R = c2ws[..., :3, :3]                       # (N,3,3)
    q = matrix_to_quaternion(R)                 # (N,4)
    q0 = q[idx0]                                # (num_new,4)
    q1 = q[idx1]
    qs = slerp(q0, q1, alpha)                   # (num_new,4)
    Rs = quaternion_to_matrix(qs)               # (num_new,3,3)

    # 拼回 c2w
    c2ws_interp = torch.zeros((num_new,3,4), device=device)
    c2ws_interp[..., :3, :3] = Rs
    c2ws_interp[..., :3,  3] = centers_interp

    return c2ws_interp, centers_interp

def color_dist_loss(img, main_clr, mask):
    main_clr = torch.tensor(main_clr).float().cuda()
    distances = torch.stack([torch.norm(img - color.view(1, 3, 1, 1)/255, dim=1) for color in main_clr], dim=1)
    min_distances, _ = torch.min(distances, dim=1)
    non_zero_mask = (mask > 0)
    min_distances_masked = min_distances[non_zero_mask[:, 0, :, :]]
    if len(min_distances_masked) > 0:
        loss = torch.mean(min_distances_masked)
    else:
        loss = torch.tensor(0.0).cuda()
    return loss


def augment_image(res_img):
    # 规范化到 [0, 1]
    res_img = res_img.squeeze(0)

    # 定义增强参数
    # brightness_factors = [0.5, 0.7, 1.3, 1.5]
    # contrast_factors = [0.5, 0.7, 1.3, 1.5]
    # resolutions = [0.5, 0.7, 1.3, 1.5]  # 相对于原始大小的缩放比例
    # rotation_angles = [15, 30, -15, -30]
    # noise_stddevs = [0.1, 0.2, 0.3, 0.4]
    # translation_factors = [0.2, -0.2]  # 移动20%
    # scaling_factors = [0.9, 0.8]  # 缩小10%, 20%
    # expanding_factors = [1.1, 1.2]  # 放大10%, 20%

    brightness_factors = [0.3, 0.6]
    contrast_factors = [0.4, 0.8]
    resolutions = [0.5, 1.5]  # 相对于原始大小的缩放比例
    rotation_angles = [30, -30]
    noise_stddevs = [0.2, 0.4]
    translation_factors = [0.2, -0.2]  # 移动20%
    # scaling_factors = [0.9, 0.8]  # 缩小10%, 20%
    # expanding_factors = [1.1, 1.2]  # 放大10%, 20%

    # 用于存储增强后的图片
    augmented_images = []
    augmented_images.append(res_img)

    # 亮度增强
    for factor in brightness_factors:
        transform = T.ColorJitter(brightness=factor)
        augmented_img = transform(res_img)
        augmented_images.append(augmented_img)

    # 对比度增强
    for factor in contrast_factors:
        transform = T.ColorJitter(contrast=factor)
        augmented_img = transform(res_img)
        augmented_images.append(augmented_img)

    # 分辨率变化
    # for scale in resolutions:
    #     height, width = res_img.shape[1:]
    #     new_height, new_width = int(height * scale), int(width * scale)
    #     transform = T.Resize((new_height, new_width), antialias=False)
    #     resized_img = transform(res_img)
    #     # transform_back = T.Resize((height, width))
    #     # resized_img_back = transform_back(resized_img)
    #     # augmented_images.append(resized_img_back)
    #     augmented_images.append(resized_img)
    #
    # # 旋转
    # for angle in rotation_angles:
    #     rotated_img = TF.rotate(res_img, angle, fill=(1,))
    #     augmented_images.append(rotated_img)
    #
    # # 高斯噪声
    # for stddev in noise_stddevs:
    #     noise = torch.randn_like(res_img).cuda() * stddev
    #     noisy_img = res_img + noise
    #     noisy_img = torch.clamp(noisy_img, 0, 1)  # 确保值在 [0, 1] 之间
    #     augmented_images.append(noisy_img)
    #
    # # 平移
    # for factor in translation_factors:
    #     # 左右移动
    #     translate = (int(width * factor), 0)
    #     translated_img = TF.affine(res_img, angle=0, translate=translate, scale=1, shear=0, fill=(1,))
    #     augmented_images.append(translated_img)
    #     # 上下移动
    #     translate = (0, int(height * factor))
    #     translated_img = TF.affine(res_img, angle=0, translate=translate, scale=1, shear=0, fill=(1,))
    #     augmented_images.append(translated_img)

    # 缩小并填充
    # for factor in scaling_factors:
    #     new_size = (int(height * factor), int(width * factor))
    #     small_img = TF.resize(res_img, new_size)
    #     padding = (width - new_size[1], height - new_size[0])
    #     pad_width = (padding[0] // 2, padding[1] // 2, padding[0] - padding[0] // 2, padding[1] - padding[1] // 2)
    #     padded_img = TF.pad(small_img, pad_width, fill=1)
    #     augmented_images.append(padded_img)

    # 放大并裁剪
    # for factor in expanding_factors:
    #     new_size = (int(height * factor), int(width * factor))
    #     large_img = TF.resize(res_img, new_size)
    #     top = (new_size[0] - height) // 2
    #     left = (new_size[1] - width) // 2
    #     cropped_img = TF.crop(large_img, top, left, height, width)
    #     augmented_images.append(cropped_img)

    # 处理增强后的图像
    processed_images = [img for img in augmented_images]

    # 可视化增强后的图像
    # fig, axes = plt.subplots(len(processed_images) // 5 + 1, 5, figsize=(15, len(processed_images) // 5 * 3))
    # for i, img in enumerate(processed_images):
    #     img_np = img.permute(1, 2, 0).detach().cpu().numpy()  # 转换为 [H, W, C] 格式并转换为 numpy 数组
    #     row, col = divmod(i, 5)
    #     axes[row, col].imshow(img_np.astype('uint8'))
    #     axes[row, col].axis('off')
    #
    # # 移除多余的子图
    # for j in range(i + 1, len(axes.flat)):
    #     axes.flat[j].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return processed_images


def compute_adv_loss(pred_bboxes, pred_scores, pred_classes, gt_bbox, target_class_idx=3):
    # 计算每个预测框与 gt_bbox 的 IoU
    ious = compute_iou(pred_bboxes, gt_bbox)  # [N]，与 gt_bbox 的 IoU

    # 筛选出类别为 target_class_idx 的预测框
    target_class_mask = (pred_classes == target_class_idx)

    # 找到所有类别为 target_class_idx 的预测框
    target_class_bboxes = pred_bboxes[target_class_mask]
    target_class_scores = pred_scores[target_class_mask]
    target_class_ious = ious[target_class_mask]

    # 如果存在类别为 target_class_idx 的预测框
    if target_class_bboxes.shape[0] > 0:
        # 找到 IoU 最大的预测框的索引
        max_iou_idx = torch.argmax(target_class_ious)
        # 取该预测框的 target_class_idx 的置信度分数作为损失
        loss = target_class_scores[max_iou_idx]
    else:
        # 如果不存在类别为 target_class_idx 的预测框，则损失为 0
        # todo min过程损失是否有问题？
        loss = torch.tensor(0.0, device=gt_bbox.device)

    # 判断攻击是否成功
    # 如果存在类别为 target_class_idx 且 IoU 大于 0.5 的检测框，则攻击不成功
    atk_success = not torch.any((target_class_ious > 0.5))

    return loss, atk_success

def compute_adv_total_loss(
    pred_bboxes: torch.Tensor,    # [N,4]
    pred_scores: torch.Tensor,    # [N]  （置信度 for each box）
    pred_classes: torch.Tensor,   # [N]  （predicted class idx）
    gt_bbox: torch.Tensor,        # [4]
    target_class_idx: int = 3,
    reg_loss_weight: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    计算对抗总损失：
      1) 检测损失 = 分类（置信度）+ 回归（L1）
    返回：
      total_loss, cls_loss, reg_loss, atk_success
    """
    # print(f"  [Loss Fn DBG] --- Entering compute_adv_total_loss ---")
    # print(f"  [Loss Fn DBG] Input pred_bboxes ({pred_bboxes.shape}):\n{pred_bboxes}")
    # print(f"  [Loss Fn DBG] Input pred_scores ({pred_scores.shape}):\n{pred_scores}")
    # print(f"  [Loss Fn DBG] Input pred_classes ({pred_classes.shape}):\n{pred_classes}")
    # print(f"  [Loss Fn DBG] Input gt_bbox: {gt_bbox}")
    
    # —— 1. 检测损失 —— #
    # IoU & 筛选目标类别
    # Handle case where no boxes are predicted
    if pred_bboxes.numel() == 0:
        # print(f"  [Loss Fn DBG] No predicted boxes. Setting losses to 0.")
        loss_device = gt_bbox.device
        total_loss = torch.tensor(0., device=loss_device)
        cls_loss = torch.tensor(0., device=loss_device)
        reg_loss = torch.tensor(0., device=loss_device)
        atk_success = True
        # print(f"  [Loss Fn DBG] --- Exiting compute_adv_total_loss (no preds) ---")
        return total_loss, cls_loss, reg_loss, atk_success

    ious = compute_iou(pred_bboxes, gt_bbox)               # [N]
    mask = (pred_classes == target_class_idx)             # [N]
    
    # print(f"  [Loss Fn DBG] Calculated IoUs (shape {ious.shape}): {ious}")
    bboxes_t = pred_bboxes[mask]                          # [M,4]
    scores_t = pred_scores[mask]                          # [M]
    ious_t   = ious[mask]                                 # [M]

    if bboxes_t.numel() > 0:
        idx = torch.argmax(ious_t)
        # 分类损失：置信度本身作为 loss（越高越“好被检测”）
        cls_loss = scores_t[idx]
        # 回归损失：L1 距离
        reg_loss = F.l1_loss(bboxes_t[idx], gt_bbox[0])
        atk_success = not (ious_t > 0.5).any().item()
    else:
        cls_loss = torch.tensor(0., device=gt_bbox.device)
        reg_loss = torch.tensor(0., device=gt_bbox.device)
        atk_success = True

    total_loss = cls_loss - reg_loss_weight * reg_loss
    # total_loss = cls_loss


    return total_loss, cls_loss, reg_loss, atk_success

################## detectron2 version #######################
# def compute_adv_loss(pred_bboxes, pred_scores, pred_classes, gt_bbox, target_class_idx=3):
#     # 计算每个预测框与 gt_bbox 的 IoU1
#     ious = compute_iou(pred_bboxes.tensor, gt_bbox)  # [N]，与 gt_bbox 的 IoU
#
#     # 筛选出类别为 target_class_idx 的预测框
#     target_class_mask = (pred_classes == target_class_idx)
#
#     # 找到所有类别为 target_class_idx 的预测框
#     target_class_bboxes = pred_bboxes.tensor[target_class_mask]
#     target_class_scores = pred_scores[target_class_mask]
#     target_class_ious = ious[target_class_mask]
#
#     # 如果存在类别为 target_class_idx 的预测框
#     if target_class_bboxes.shape[0] > 0:
#         # 找到 IoU 最大的预测框的索引
#         max_iou_idx = torch.argmax(target_class_ious)
#         # 取该预测框的 target_class_idx 的置信度分数作为损失
#         loss = target_class_scores[max_iou_idx]
#     else:
#         # 如果不存在类别为 target_class_idx 的预测框，则损失为 0
#         loss = torch.tensor(0.0, device=gt_bbox.device)
#
#     # 判断攻击是否成功
#     # 如果存在类别为 target_class_idx 且 IoU 大于 0.5 的检测框，则攻击不成功
#     atk_success = not torch.any((target_class_ious > 0.5))
#
#     return loss, atk_success

def compute_iou(bboxes1, bbox2):
    """
    计算每个预测框与给定gt_bbox的IoU。

    参数：
    - bboxes1 (Tensor): 预测边界框，形状为 [N, 4]。
    - bbox2 (Tensor): ground truth 边界框，形状为 [4]。

    返回：
    - ious (Tensor): IoU，形状为 [N]。
    """
    bbox2 = bbox2.squeeze(0)

    # 计算交集
    x1 = torch.max(bboxes1[:, 0], bbox2[0])
    y1 = torch.max(bboxes1[:, 1], bbox2[1])
    x2 = torch.min(bboxes1[:, 2], bbox2[2])
    y2 = torch.min(bboxes1[:, 3], bbox2[3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算每个框的面积
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算并集
    union_area = area1 + area2 - inter_area

    # 计算IoU
    ious = inter_area / torch.clamp(union_area, min=1e-6)

    return ious

def save_dark_pixels_mask(img, save_dir, id, threshold=(10 / 255, 10 / 255, 10 / 255)):
    threshold_tensor = torch.tensor(threshold).view(3, 1, 1).float().cuda()
    mask = (img < threshold_tensor).all(dim=0)
    mask_np = mask.cpu().numpy()
    save_path = f"{save_dir}/results_view_{id}"
    np.save(save_path, mask_np)


def add_inf_mask(img, mask_path):
    mask = np.load(mask_path)
    mask_tensor = torch.tensor(mask).float().unsqueeze(0)  # 形状为 (1, height, width)
    mask_tensor = mask_tensor.expand_as(img)
    white_tensor = torch.ones_like(img)
    img[mask_tensor == 1] = img[mask_tensor == 1] + white_tensor[mask_tensor == 1]
    modified_img_tensor = torch.clamp(img, 0.0, 1.0)
    return modified_img_tensor


def visualize_predictions_from_tensor(input_tensor, preds, view, i, score_threshold=0.5,
                                      output_dir='./vis_detect_res/'):
    """
    对预测结果进行可视化，并将结果保存到指定文件。

    参数:
    - input_tensor: torch.Tensor, 输入图像的张量，形状为 [C, H, W]。
    - preds: list, Faster R-CNN 的预测结果。
    - view: int, 视角编号，用于保存文件命名。
    - i: int, 图像编号，用于保存文件命名。
    - score_threshold: float, 置信度阈值，仅绘制高于该阈值的检测框。
    - output_dir: str, 输出目录，保存检测结果。
    """
    # 将图像张量转换为 PIL 图像
    to_pil = transforms.ToPILImage()
    if isinstance(input_tensor, list):
        input_tensor = input_tensor[0]
    image = to_pil(input_tensor.squeeze(0).cpu())

    # 创建 ImageDraw 对象来在图像上绘制边界框和标签
    draw = ImageDraw.Draw(image)

    # 获取第一个预测（假设只有一个输入）
    pred = preds[0]
    boxes = pred['boxes'].detach().cpu().numpy()
    scores = pred['scores'].detach().cpu().numpy()
    labels = pred['labels'].detach().cpu().numpy()

    # 使用默认字体
    font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            # 绘制边界框
            draw.rectangle(box.tolist(), outline="red", width=3)
            # 获取类别名称
            class_name = coco_classes[label] if label < len(coco_classes) else "Unknown"
            # 添加标签和置信度
            text = f"{class_name}: {score:.2f}"
            # 计算文本大小
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_location = [box[0], box[1] - text_height]
            if text_location[1] < 0:
                text_location[1] = box[1]  # 防止标签在图像框外

            draw.rectangle(
                [text_location[0], text_location[1], text_location[0] + text_width, text_location[1] + text_height],
                fill="red"
            )
            draw.text((text_location[0], text_location[1]), text, fill="white", font=font)

    # 保存绘制的图像
    output_path = f'{output_dir}results_view{view}_{i}_detected.png'
    image.save(output_path)
    # print(f"可视化的检测结果已保存到: {output_path}")


def perturb_camera(c2w: torch.Tensor,
                   center: torch.Tensor,
                   max_angle_deg: float = 1.0,
                   max_trans: float = 0.01) -> (torch.Tensor, torch.Tensor):
    """
    对单个相机外参做小幅度随机扰动。

    参数：
      c2w           [3,4] Tensor，前三列是旋转矩阵 R，最后一列是平移向量 t
      center        [3]   Tensor，相机中心（世界坐标）
      max_angle_deg 最大旋转扰动角度（度）
      max_trans     最大平移扰动幅度（世界单位）

    返回：
      c2w_new       [3,4] Tensor，扰动后的外参
      center_new    [3]   Tensor，扰动后的相机中心
    """
    device = c2w.device

    # 拆分 R 和 t
    R_orig = c2w[:, :3]    # [3,3]
    t_orig = c2w[:, 3]     # [3]

    # 1) 随机旋转轴与角度
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm(p=2)
    angle = (torch.rand(1, device=device) * 2 - 1) * (max_angle_deg * math.pi / 180)

    # Rodrigues 叉乘矩阵
    K = torch.tensor([[    0, -axis[2],  axis[1]],
                      [ axis[2],      0, -axis[0]],
                      [-axis[1],  axis[0],     0]],
                     device=device)
    R_noise = torch.eye(3, device=device) \
              + torch.sin(angle) * K \
              + (1 - torch.cos(angle)) * (K @ K)

    # 2) 随机平移
    t_noise = (torch.rand(3, device=device) * 2 - 1) * max_trans

    # 3) 应用扰动到 R 和 t
    R_new = R_noise @ R_orig            # [3,3]
    t_new = R_noise @ t_orig + t_noise  # [3]

    # 4) 计算新的 center
    center_new = R_noise @ center + t_noise

    # 5) 拼回 [3,4]
    c2w_new = torch.cat([R_new, t_new.unsqueeze(1)], dim=1)  # [3,4]

    return c2w_new, center_new

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side1, patch_side2):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side1, patch_side2),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side1, side2):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side1, side2), red))
            printability_imgs.append(np.full((side1, side2), green))
            printability_imgs.append(np.full((side1, side2), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


# =================================================================================
# mAP Calculation Logic (manual implementation)
# =================================================================================

def bbox_iou(bboxes1, bboxes2, eps=1e-6):
    """Calculate the Intersection over Union (IoU) between two sets of bboxes.
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    inter_area = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    
    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    union_area = box1_area + np.transpose(box2_area) - inter_area
    
    iou = inter_area / (union_area + eps)
    return iou


def tpfp_default(det_bboxes, gt_bboxes, iou_thr=0.5):
    """Check if detected bboxes are true positive or false positive."""
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if num_gts == 0:
        return np.zeros(num_dets, dtype=np.int8), np.zeros(num_dets, dtype=np.int8)

    ious = bbox_iou(det_bboxes[:, :4], gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    
    tp = np.zeros(num_dets, dtype=np.int8)
    fp = np.zeros(num_dets, dtype=np.int8)
    
    gt_covered = np.zeros(num_gts, dtype=bool)

    # sort all detections by confidence
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for i in sort_inds:
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
            
    return tp, fp


def eval_map(det_results, gt_bboxes, num_classes, iou_thr=0.5):
    """Evaluate mAP of a dataset."""
    all_cls_aps = []
    for i in range(num_classes):
        # 1. Get predictions and GTs for this class
        cls_dets = np.vstack([res[i] for res in det_results]) if any(res[i].size > 0 for res in det_results) else np.empty((0, 5))
        cls_gts = np.vstack([gt['bboxes'] for gt in gt_bboxes if i in gt['labels']]) if any(i in gt['labels'] for gt in gt_bboxes) else np.empty((0, 4))
        
        if cls_dets.shape[0] == 0:
            if cls_gts.shape[0] > 0:
                all_cls_aps.append(0.0) # Detections missed
            else:
                all_cls_aps.append(1.0) # No GTs, no detections, perfect score
            continue

        # 2. Sort detections by confidence
        sort_inds = np.argsort(-cls_dets[:, -1])
        cls_dets = cls_dets[sort_inds, :]
        
        # 3. Calculate TP/FP
        tp, fp = tpfp_default(cls_dets, cls_gts, iou_thr)
        
        # 4. Calculate precision and recall
        tp_cumsum = np.cumsum(tp).astype(np.float32)
        fp_cumsum = np.cumsum(fp).astype(np.float32)
        
        eps = np.finfo(np.float32).eps
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        recalls = tp_cumsum / (cls_gts.shape[0] if cls_gts.shape[0] > 0 else 1)
        
        # 5. Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.
        all_cls_aps.append(ap)
        
    mean_ap = np.mean(all_cls_aps)
    return {'AP50': mean_ap}


def mean_ap(det_results, gt_bboxes_list, iou_thr=0.5, num_classes=80, nproc=1):
    # This is a wrapper to match the expected signature. nproc is ignored.
    return eval_map(det_results, gt_bboxes_list, num_classes=num_classes, iou_thr=iou_thr)


def calculate_ap_for_target_class(det_results, gt_bboxes_list, target_class_idx, iou_thr=0.5):
    """
    Calculates Average Precision (AP) for a single target class, 
    assuming one ground truth object per image.
    """
    # 1. Filter predictions and GTs for the target class
    cls_dets = []
    for i, res in enumerate(det_results):
        if res[target_class_idx].size > 0:
            for det in res[target_class_idx]:
                # Store [bbox, score, image_idx]
                cls_dets.append(np.append(det, i))
    
    if not cls_dets:
        # If no detections for the target class, AP is 0 unless there are also no GTs.
        has_gts = any(target_class_idx in gt['labels'] for gt in gt_bboxes_list)
        return {'AP50': 0.0 if has_gts else 1.0}
        
    cls_dets = np.array(cls_dets)

    # All GTs are for the target class in this context
    num_gts = len(gt_bboxes_list)

    # 2. Sort detections by confidence
    sort_inds = np.argsort(-cls_dets[:, 4])
    cls_dets = cls_dets[sort_inds, :]
    
    tp = np.zeros(cls_dets.shape[0])
    fp = np.zeros(cls_dets.shape[0])
    
    # Keep track of matched GTs by their image index
    gt_matched = [False] * len(gt_bboxes_list)
    
    for i in range(cls_dets.shape[0]):
        det = cls_dets[i, :]
        img_idx = int(det[5])
        
        if gt_matched[img_idx]:
            fp[i] = 1
            continue
            
        gt_bbox = gt_bboxes_list[img_idx]['bboxes']
        
        iou = bbox_iou(det[:4].reshape(1, 4), gt_bbox)
        
        if iou >= iou_thr:
            tp[i] = 1
            gt_matched[img_idx] = True
        else:
            fp[i] = 1
            
    # 4. Calculate precision and recall
    tp_cumsum = np.cumsum(tp).astype(np.float32)
    fp_cumsum = np.cumsum(fp).astype(np.float32)
    
    eps = np.finfo(np.float32).eps
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
    recalls = tp_cumsum / num_gts if num_gts > 0 else 0
    
    # 5. Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.
        
    return {'AP50': ap}