# -*- coding: utf-8 -*-
"""
本脚本定义了 LBMRelighter 类，用于实现基于物理渲染的背景重打光（Relighting）。

功能概述:
- 模型加载：从指定的检查点目录加载预训练的 LBM (Light-Bearing Manifold) 模型。
- 图像处理：提供 `relight` 方法，接收原始背景图、前景重打光图（如3D高斯渲染的白底车辆）和背景蒙版。
- 智能拼接：
    1. 使用蒙版将原始背景中的物体（如车辆）擦除。
    2. 对擦除后的背景进行修复（Inpainting），填补空洞。
    3. 将修复后的背景与提供的前景重打光图智能地拼接在一起，生成光照一致的最终图像。
- 应用场景：在自动驾驶仿真或数据增强中，用于将虚拟物体（如应用了对抗纹理的车辆）真实地融入到任意背景图片中，同时保持光照的物理正确性。
"""

import logging
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

# It's better to import from the lbm library directly if it's installed
# If not, ensure the path to lbm is in PYTHONPATH
try:
    from lbm.inference import get_model
except ImportError:
    # Add lbm to path if it's not installed, adjust as needed
    import sys
    sys.path.append('/workspace/lbm/src')
    from lbm.inference import get_model


logging.basicConfig(level=logging.INFO)


class LBMRelighter:
    """
    A class to handle LBM model loading and background relighting inference.
    """

    def __init__(self, ckpt_dir: str, device: str = "cuda", torch_dtype=torch.bfloat16):
        """
        Initializes the relighter and loads the model.

        Args:
            ckpt_dir (str): Path to the folder containing config.yaml and .ckpt file.
            device (str): The device to run the model on ('cuda' or 'cpu').
            torch_dtype: The torch data type to use for the model.
        """
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        logging.info("Loading LBM model from checkpoint directory...")
        self.model = get_model(
            ckpt_dir,
            save_dir=None,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logging.info("LBM model loaded successfully.")
        self.to_tensor = ToTensor()

    def _load_rgb(self, path: str) -> Image.Image:
        """Loads an image and converts it to RGB."""
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: str, size: tuple[int, int]) -> np.ndarray:
        """Loads a mask, resizes, and binarizes it."""
        m = Image.open(path).convert("L").resize(size, Image.NEAREST)
        m_np = np.array(m).astype(np.float32) / 255.0
        m_np = (m_np > 0.5).astype(np.float32)  # binarize to 0 or 1
        return m_np

    def relight(
        self,
        source_image: Image.Image,
        fg_relight_image: Image.Image,
        bg_mask: Image.Image,
        width: int = 640,
        height: int = 480,
        num_inference_steps: int = 1,
        invert_mask: bool = False,
    ) -> Image.Image:
        """
        Performs one-step background relighting inference.

        Args:
            source_image (Image.Image): The source image as a PIL Image object.
            fg_relight_image (Image.Image): The foreground relight image as a PIL Image object.
            bg_mask (Image.Image): The background mask (background=1, foreground=0) as a PIL Image object.
            width (int): Inference width.
            height (int): Inference height.
            num_inference_steps (int): Number of steps for the sampler.
            invert_mask (bool): If True, inverts the mask (0 becomes 1, 1 becomes 0).

        Returns:
            Image.Image: The composed, relighted image at the original source resolution.
        """
        inference_size = (width, height)

        # 1. Prepare images from input objects
        src_orig = source_image.convert("RGB")
        orig_size = src_orig.size
        src = src_orig.resize(inference_size, Image.BILINEAR)
        fg_relight = fg_relight_image.convert("RGB").resize(inference_size, Image.BILINEAR)

        # Convert mask PIL to numpy array, resize and binarize
        mask_pil = bg_mask.convert("L").resize(inference_size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)  # binarize to 0 or 1
        
        if invert_mask:
            mask_np = 1.0 - mask_np
            
        src_np = np.array(src, dtype=np.float32) / 255.0
        fg_relight_np = np.array(fg_relight, dtype=np.float32) / 255.0
        
        mask_3d = np.stack([mask_np] * 3, axis=-1)
        fg_mask_3d = 1.0 - mask_3d
        
        composed_fg_np = (fg_mask_3d * fg_relight_np + mask_3d * src_np).clip(0, 1)
        composed_fg_pil = Image.fromarray((composed_fg_np * 255.0).astype(np.uint8))

        # 2. Build batch for model
        src_t = (self.to_tensor(src) * 2 - 1).unsqueeze(0).to(self.device, self.torch_dtype)
        fg_t = (self.to_tensor(composed_fg_pil) * 2 - 1).unsqueeze(0).to(self.device, self.torch_dtype)
        
        batch = {
            self.model.source_key: src_t,
            "fg_relight": fg_t,
        }

        # 3. Run model inference
        with torch.no_grad():
            z_source = self.model.vae.encode(batch[self.model.source_key])
            out_tensor = self.model.sample(
                z=z_source,
                num_steps=num_inference_steps,
                conditioner_inputs=batch,
                max_samples=1,
            ).clamp(-1, 1)
        
        # Decode output to PIL image
        out_img_tensor = (out_tensor[0].float().cpu() + 1) / 2
        out_pil = Image.fromarray((out_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        # Ensure model output size matches inference size, as some models might produce slightly different dimensions
        if out_pil.size != inference_size:
            out_pil = out_pil.resize(inference_size, Image.BILINEAR)

        # 4. Post-process to create 'output_composed_origsize'
        # We use the original fg_relight image for the foreground composition
        # as it's cleaner than the one composed with the downsampled source bg.
        fg_relight_orig = fg_relight_image.convert("RGB")
        
        # The mask needs to be at the inference resolution for composition
        out_np = np.array(out_pil, dtype=np.float32) / 255.0
        fg_relight_resized_np = np.array(fg_relight, dtype=np.float32) / 255.0

        # print('mask_3d.shape:', mask_3d.shape, 'out_np.shape:', out_np.shape, 'fg_mask_3d.shape:', fg_mask_3d.shape, 'fg_relight_resized_np.shape:', fg_relight_resized_np.shape)
        composed_np = mask_3d * out_np + fg_mask_3d * fg_relight_resized_np
        composed_np = (composed_np.clip(0, 1) * 255.0).astype(np.uint8)
        composed_pil = Image.fromarray(composed_np)

        # Resize to original source image size
        final_image = composed_pil.resize(orig_size, Image.LANCZOS)
        
        return final_image

    def relight_hdr(
        self,
        source_image: Image.Image,
        fg_relight_image: Image.Image,
        bg_mask: Image.Image,
        width: int = 640,
        height: int = 480,
        num_inference_steps: int = 1,
        invert_mask: bool = False,
        hdr_sh_coeffs: torch.Tensor | None = None,
    ) -> Image.Image:
        """
        Performs one-step background relighting inference with HDR conditioning.

        Args:
            source_image (Image.Image): The source image as a PIL Image object.
            fg_relight_image (Image.Image): The foreground relight image as a PIL Image object.
            bg_mask (Image.Image): The background mask (background=1, foreground=0) as a PIL Image object.
            width (int): Inference width.
            height (int): Inference height.
            num_inference_steps (int): Number of steps for the sampler.
            invert_mask (bool): If True, inverts the mask (0 becomes 1, 1 becomes 0).
            hdr_sh_coeffs (torch.Tensor | None): Optional tensor of SH coefficients for HDR relighting.

        Returns:
            Image.Image: The composed, relighted image at the original source resolution.
        """
        print("Performing HDR relighting...")
        inference_size = (width, height)

        # 1. Prepare images from input objects
        src_orig = source_image.convert("RGB")
        orig_size = src_orig.size
        src = src_orig.resize(inference_size, Image.BILINEAR)
        fg_relight = fg_relight_image.convert("RGB").resize(inference_size, Image.BILINEAR)

        # Convert mask PIL to numpy array, resize and binarize
        mask_pil = bg_mask.convert("L").resize(inference_size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)  # binarize to 0 or 1
        
        if invert_mask:
            mask_np = 1.0 - mask_np
            
        src_np = np.array(src, dtype=np.float32) / 255.0
        fg_relight_np = np.array(fg_relight, dtype=np.float32) / 255.0
        
        mask_3d = np.stack([mask_np] * 3, axis=-1)
        fg_mask_3d = 1.0 - mask_3d
        
        composed_fg_np = (fg_mask_3d * fg_relight_np + mask_3d * src_np).clip(0, 1)
        composed_fg_pil = Image.fromarray((composed_fg_np * 255.0).astype(np.uint8))

        # 2. Build batch for model
        src_t = (self.to_tensor(src) * 2 - 1).unsqueeze(0).to(self.device, self.torch_dtype)
        fg_t = (self.to_tensor(composed_fg_pil) * 2 - 1).unsqueeze(0).to(self.device, self.torch_dtype)
        
        batch = {
            self.model.source_key: src_t,
            "fg_relight": fg_t,
        }


        C = hdr_sh_coeffs.shape[0]
        H, W = height, width
        # Reshape and broadcast to match image dimensions [1, C, H, W]
        hdr_t = hdr_sh_coeffs.view(C, 1, 1).expand(C, H, W).unsqueeze(0).to(self.device, self.torch_dtype)
        batch["hdr"] = hdr_t


        # 3. Run model inference
        with torch.no_grad():
            z_source = self.model.vae.encode(batch[self.model.source_key])
            out_tensor = self.model.sample(
                z=z_source,
                num_steps=num_inference_steps,
                conditioner_inputs=batch,
                max_samples=1,
            ).clamp(-1, 1)
        
        # Decode output to PIL image
        out_img_tensor = (out_tensor[0].float().cpu() + 1) / 2
        out_pil = Image.fromarray((out_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        # Ensure model output size matches inference size, as some models might produce slightly different dimensions
        if out_pil.size != inference_size:
            out_pil = out_pil.resize(inference_size, Image.BILINEAR)

        # 4. Post-process to create 'output_composed_origsize'
        # We use the original fg_relight image for the foreground composition
        # as it's cleaner than the one composed with the downsampled source bg.
        fg_relight_orig = fg_relight_image.convert("RGB")
        
        # The mask needs to be at the inference resolution for composition
        out_np = np.array(out_pil, dtype=np.float32) / 255.0
        fg_relight_resized_np = np.array(fg_relight, dtype=np.float32) / 255.0

        # print('mask_3d.shape:', mask_3d.shape, 'out_np.shape:', out_np.shape, 'fg_mask_3d.shape:', fg_mask_3d.shape, 'fg_relight_resized_np.shape:', fg_relight_resized_np.shape)
        composed_np = mask_3d * out_np + fg_mask_3d * fg_relight_resized_np
        composed_np = (composed_np.clip(0, 1) * 255.0).astype(np.uint8)
        composed_pil = Image.fromarray(composed_np)

        # Resize to original source image size
        final_image = composed_pil.resize(orig_size, Image.LANCZOS)
        
        return final_image

if __name__ == '__main__':
    # Example usage based on your command
    
    # Define paths
    ckpt_dir = '/workspace/lbm/checkpoints'
    source_image_path = '/workspace/lbm/fg_relit_eval_input/ori/00001.jpg'
    fg_relight_image_path = '/workspace/lbm/fg_relit_eval_input/no_sun/ours_60000/renders/00000.png'
    bg_mask_path = '/workspace/lbm/fg_relit_eval_input/masks/00001_mask.png'
    output_dir = '/workspace/lbm/output_eval'
    
    # Load images before calling the function
    print("Loading images from disk...")
    source_img = Image.open(source_image_path)
    fg_relight_img = Image.open(fg_relight_image_path)
    bg_mask_img = Image.open(bg_mask_path)

    # Create the relighter instance
    relighter = LBMRelighter(ckpt_dir=ckpt_dir)
    
    # Run the relighting process
    print("Running relighting inference...")
    result_image = relighter.relight(
        source_image=source_img,
        fg_relight_image=fg_relight_img,
        bg_mask=bg_mask_img,
        num_inference_steps=1,
        invert_mask=True  # Example of using the new option
    )
    
    # Save the final image
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'lbm_relit_output.png')
    result_image.save(output_path)
    
    print(f"Relighting complete. Output saved to: {output_path}")
