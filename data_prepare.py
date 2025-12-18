import os
import sys
import argparse
import cv2
import numpy as np
import json
import torch
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("错误：无法导入 'segment_anything'。")
    print("请确认已在您的 rga3 conda 环境中安装此库：pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

# 将项目根目录（RGA）添加到 sys.path，以便正确导入 data_prep 模块
# 这假设脚本从 RGA 目录或其父目录运行
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir # 假设 /RGA/data_prepare.py，那么 script_dir 就是 /RGA
# 如果脚本在 /RGA 下，为了导入 data.data_prep，RGA 的父目录需要被添加到 sys.path
# 或者在运行时，python path 已经包含了 RGA 目录
# 为了稳健性，我们直接假设运行目录是 RGA
try:
    from data.data_prep.extract_red_mask import create_red_pink_mask_rgb
except ImportError:
    print("错误：无法从 'data.data_prep.extract_red_mask' 导入 'create_red_pink_mask_rgb'。")
    print("请确认该文件存在，并且您是从 RGA 项目的根目录运行此脚本。")
    sys.exit(1)


def process_dataset(dataset_name):
    """
    处理指定的数据集：分割车辆、提取到白色背景并生成红色掩码。
    """
    print(f"正在处理数据集: {dataset_name}")
    base_dir = os.path.join('data', dataset_name)
    ori_dir = os.path.join(base_dir, 'ori')
    annos_dir = os.path.join(base_dir, 'annos')

    # 步骤 1: 检查 ori 和 annos 文件夹是否存在
    if not os.path.isdir(ori_dir) or not os.path.isdir(annos_dir):
        print(f"错误: 在 '{base_dir}' 中未找到 'ori' 和 'annos' 文件夹。请检查路径。")
        return

    # 创建输出目录
    masks_dir = os.path.join(base_dir, 'masks')
    images_dir = os.path.join(base_dir, 'images')
    input_dir = os.path.join(base_dir, 'input')
    red_masks_dir = os.path.join(base_dir, 'red_masks')

    for d in [masks_dir, images_dir, input_dir, red_masks_dir]:
        os.makedirs(d, exist_ok=True)

    # --- 阶段 1: 加载 SAM 模型并分割车辆 ---
    print("\n阶段 1/3: 使用 SAM 分割车辆...")
    
    sam_checkpoint_path = "m_envs/sam_ckpt"
    # 查找任何 .pth 文件作为 SAM 权重
    pth_files = [f for f in os.listdir(sam_checkpoint_path) if f.endswith('.pth')]
    if not pth_files:
        print(f"错误: 在 '{sam_checkpoint_path}' 文件夹下未找到任何 .pth 权重文件。")
        sys.exit(1)
    
    sam_checkpoint = os.path.join(sam_checkpoint_path, pth_files[0])
    print(f"找到并使用 SAM 权重: {sam_checkpoint}")
    
    model_type = "vit_h" # 假设使用 vit_h 模型，如果使用其他模型请修改
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_files = sorted([f for f in os.listdir(ori_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for filename in tqdm(image_files, desc="分割图像"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(ori_dir, filename)
        anno_path = os.path.join(annos_dir, base_name + '.json')
        mask_path = os.path.join(masks_dir, base_name + '_mask.png')

        if not os.path.exists(anno_path):
            print(f"警告: 找不到图像 '{filename}' 对应的标注文件, 已跳过。")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 '{image_path}', 已跳过。")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        try:
            with open(anno_path, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            # 此处假设标注文件 (如 LabelMe) 的格式，可能需要根据您的实际格式进行调整
            if "shapes" in anno_data and len(anno_data["shapes"]) > 0:
                # 假设第一个形状是车辆的矩形框
                points = anno_data["shapes"][0]["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                input_box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            else:
                print(f"警告: 在 '{anno_path}' 中找不到有效的 'shapes' 数据, 已跳过。")
                continue
        except Exception as e:
            print(f"警告: 解析标注文件 '{anno_path}' 失败: {e}, 已跳过。")
            continue

        masks, _, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # 将分割出的掩码保存为图像文件
        final_mask = (masks[0] * 255).astype(np.uint8)
        cv2.imwrite(mask_path, final_mask)

    print("车辆分割完成。")

    # --- 阶段 2: 根据掩码提取车辆，背景置为纯白 ---
    print("\n阶段 2/3: 提取车辆并设置白色背景...")
    for filename in tqdm(image_files, desc="提取车辆"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(ori_dir, filename)
        mask_path = os.path.join(masks_dir, base_name + '_mask.png')
        
        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # 创建一个纯白色的背景
        white_bg = np.full(image.shape, 255, dtype=np.uint8)
        
        # 使用掩码提取前景和背景
        inv_mask = cv2.bitwise_not(mask)
        foreground = cv2.bitwise_and(image, image, mask=mask)
        background = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        
        # 合成最终图像
        extracted_image = cv2.add(foreground, background)
        
        # 保存到 images 和 input 文件夹
        cv2.imwrite(os.path.join(images_dir, filename), extracted_image)
        cv2.imwrite(os.path.join(input_dir, filename), extracted_image)

    print("车辆提取完成。")

    # --- 阶段 3: 利用 data_prep 中的脚本提取红色掩码 ---
    print("\n阶段 3/3: 提取红色区域掩码...")
    create_red_pink_mask_rgb(input_dir, red_masks_dir)
    print("红色掩码提取完成。")
    print(f"\n数据集 '{dataset_name}' 处理完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 RGA 准备数据集：分割、提取车辆，并生成红色掩码。")
    parser.add_argument("--dataset_name", type=str, default="rga", help="位于 'data/' 目录下的数据集文件夹名称。")
    args = parser.parse_args()
    
    process_dataset(args.dataset_name)
