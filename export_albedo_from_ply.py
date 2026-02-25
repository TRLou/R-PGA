import argparse
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render


@torch.no_grad()
def export_albedo_views(args: argparse.Namespace) -> None:
    ckpt_path = getattr(args, "checkpoint_path", "")
    ckpt_path = str(ckpt_path) if ckpt_path is not None else ""
    ckpt_file = Path(ckpt_path) if ckpt_path else None
    if ckpt_file and not ckpt_file.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_file}")

    ply_path = Path(args.ply_path)
    if (ckpt_file is None) and (not ply_path.is_file()):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset/cameras
    model_params = ModelParams(argparse.ArgumentParser())
    model_params = model_params.extract(args)
    pipe = PipelineParams(argparse.ArgumentParser()).extract(args)
    
    # 如果 source_path 为空，尝试从 model_path 的 cfg_args 读取
    if not getattr(model_params, "source_path", None) or model_params.source_path is None:
        if hasattr(args, "model_path") and args.model_path:
            cfg_path = Path(args.model_path) / "cfg_args"
            if cfg_path.is_file():
                try:
                    with open(cfg_path, 'r') as f:
                        cfg_string = f.read()
                    cfg_args = eval(cfg_string)
                    if hasattr(cfg_args, "source_path") and cfg_args.source_path:
                        model_params.source_path = cfg_args.source_path
                        print(f"[消息] 从 cfg_args 读取 source_path: {model_params.source_path}")
                except Exception as e:
                    print(f"[警告] 读取 cfg_args 失败: {e}")
    
    if not getattr(model_params, "source_path", None):
        raise ValueError("请提供 --source_path 指向数据集根目录，或提供 --model_path 以便从 cfg_args 读取。")
    if not Path(model_params.source_path).is_dir():
        raise FileNotFoundError(f"source_path 不存在: {model_params.source_path}")

    # 处理 environment_texture：空字符串应转为 None
    env_texture = getattr(args, "environment_texture", "")
    if env_texture == "" or env_texture is None:
        env_texture = None
    elif not Path(env_texture).is_file():
        print(f"[警告] environment_texture 文件不存在: {env_texture}，将使用 None")
        env_texture = None

    # Create scene to load cameras; gaussians will be overwritten by ply
    gaussians = GaussianModel(
        model_params.sh_degree,
        environment_texture=env_texture,
        environment_scale=float(getattr(args, "environment_scale", 1.0)),
    )
    scene = Scene(model_params, gaussians, load_iteration=None, shuffle=False)

    # Load model state: prefer checkpoint if provided
    loaded_from_ckpt = False

    if ckpt_file is not None:
        print(f"[消息] 正在加载 checkpoint: {ckpt_file}")
        try:
            ckpt_data = torch.load(str(ckpt_file), map_location="cuda")
            # Expected format: (model_tuple, iteration)
            if isinstance(ckpt_data, (tuple, list)) and len(ckpt_data) >= 1:
                model_tuple = ckpt_data[0]
                if isinstance(model_tuple, (tuple, list)) and len(model_tuple) >= 10:
                    # Follow GaussianModel.capture() layout
                    gaussians.active_sh_degree = int(model_tuple[0])
                    gaussians._xyz = model_tuple[1]
                    gaussians._features_dc = model_tuple[2]
                    gaussians._features_rest = model_tuple[3]
                    gaussians._scaling = model_tuple[4]
                    gaussians._rotation = model_tuple[5]
                    gaussians._opacity = model_tuple[6]
                    gaussians._albedo_init = model_tuple[7]
                    gaussians._metallic_init = model_tuple[8]
                    gaussians._roughness_init = model_tuple[9]
                    if len(model_tuple) > 10:
                        gaussians.diffuse_occ = model_tuple[10]
                    if len(model_tuple) > 11:
                        gaussians.grid = model_tuple[11]
                    if len(model_tuple) > 12:
                        gaussians.max_pts = model_tuple[12]
                    if len(model_tuple) > 13:
                        gaussians.min_pts = model_tuple[13]
                    if len(model_tuple) > 14:
                        gaussians.max_radii2D = model_tuple[14]
                    if len(model_tuple) > 19 and isinstance(model_tuple[19], dict):
                        gaussians.envlight.load_state_dict(model_tuple[19])
                    loaded_from_ckpt = True
                    print(f"[消息] 已从 checkpoint 加载模型: {ckpt_file}")
        except Exception as e:
            print(f"[警告] 读取 checkpoint 失败: {e}，将回退到 PLY")

    if not loaded_from_ckpt:
        # Load ply into gaussians and enable full SH
        gaussians.load_ply(str(ply_path))
        gaussians.active_sh_degree = gaussians.max_sh_degree
        # 若未显式提供 envlight，则尝试从 model_path 的最新 checkpoint 加载光照
        if env_texture is None and hasattr(args, "model_path") and args.model_path:
            model_dir = Path(args.model_path)
            if model_dir.is_dir():
                ckpts = []
                for p in model_dir.glob("chkpnt*.pth"):
                    try:
                        num = int(p.stem.replace("chkpnt", ""))
                        ckpts.append((num, p))
                    except Exception:
                        continue
                if ckpts:
                    ckpts.sort(key=lambda x: x[0])
                    latest_ckpt = ckpts[-1][1]
                    try:
                        ckpt_data = torch.load(str(latest_ckpt), map_location="cuda")
                        if isinstance(ckpt_data, (tuple, list)) and len(ckpt_data) >= 1:
                            model_tuple = ckpt_data[0]
                            if isinstance(model_tuple, (tuple, list)) and len(model_tuple) > 19 and isinstance(model_tuple[19], dict):
                                gaussians.envlight.load_state_dict(model_tuple[19])
                                print(f"[消息] 已从最新 checkpoint 加载 envlight: {latest_ckpt}")
                    except Exception as e:
                        print(f"[警告] 读取最新 checkpoint envlight 失败: {e}")
    
    # 初始化体素网格和遮挡信息（第二阶段渲染需要）
    # get_diffuse_occ() 内部会调用 get_grid()，从而初始化 min_pts 和 max_pts
    render_iteration = int(args.render_iteration)
    if render_iteration < 0:
        render_iteration = int(args.second_stage_step) + 1
    
    if render_iteration > int(args.second_stage_step) and (not loaded_from_ckpt or gaussians.diffuse_occ.numel() == 0):
        try:
            print("[消息] 初始化体素网格和遮挡信息（第二阶段渲染需要）...")
            gaussians.get_diffuse_occ()
        except Exception as e:
            print(f"[警告] 初始化遮挡信息失败: {e}，将尝试仅初始化网格...")
            try:
                gaussians.get_grid()
            except Exception as e2:
                print(f"[警告] 初始化体素网格也失败: {e2}")
    
    try:
        # 确保 envlight 完整可用（生成 base 与 mips）
        gaussians.envlight.build_base()
        gaussians.envlight.build_mips()
    except Exception:
        pass

    # Background color
    bg_color = [1.0, 1.0, 1.0] if model_params.white_background else [0.0, 0.0, 0.0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Choose split
    split = str(args.split).lower()
    if split == "train":
        cameras = scene.getTrainCameras()
    elif split == "test":
        cameras = scene.getTestCameras()
    else:
        cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())

    if args.max_views and int(args.max_views) > 0:
        cameras = cameras[: int(args.max_views)]

    print(f"[消息] 将导出 {len(cameras)} 个视角的 albedo 到: {out_dir}")
    for cam in tqdm(cameras, desc="Export albedo", ncols=120):
        render_pkg = render(
            cam,
            gaussians,
            pipe,
            bg,
            iteration=render_iteration,
            is_train=False,
            first_stage_step=int(args.first_stage_step),
            second_stage_step=int(args.second_stage_step),
            hdr_rotation=bool(args.hdr_rotation),
        )
        rendered_albedo = render_pkg["rendered_albedo"]
        if rendered_albedo is None:
            raise RuntimeError(
                "rendered_albedo is None. "
                "请设置 --render_iteration 为大于 second_stage_step 的值。"
            )
        save_path = out_dir / f"{cam.image_name}.png"
        torchvision.utils.save_image(rendered_albedo.clamp(0.0, 1.0).detach().cpu(), str(save_path))

    print("[消息] albedo 导出完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 PLY 导出各视角 albedo")

    # Reuse existing param groups for dataset/pipeline
    ModelParams(parser)
    PipelineParams(parser)

    parser.add_argument("--ply_path", type=str, default="RGA_output/0201_143919_Beijing/point_cloud_final.ply", help="指向 point_cloud_final.ply 的路径")
    parser.add_argument("--output_dir", type=str, default="RGA_output/0201_143919_Beijing/albedo_views", help="albedo 输出目录")
    parser.add_argument("--checkpoint_path", type=str, default="", help="可选：加载 checkpoint（.pth）代替 PLY")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"], help="导出视角集合")
    parser.add_argument("--max_views", type=int, default=-1, help="最多导出视角数（<=0 表示不限制）")
    parser.add_argument("--render_iteration", type=int, default=60000, help="渲染时使用的 iteration（<0 表示 second_stage_step+1）")
    parser.add_argument("--first_stage_step", type=int, default=5000, help="与训练一致的第一阶段阈值")
    parser.add_argument("--second_stage_step", type=int, default=30000, help="与训练一致的第二阶段阈值")
    parser.add_argument("--hdr_rotation", action="store_true", default=False, help="与训练一致 HDR 旋转开关")
    parser.add_argument("--environment_texture", type=str, default="", help="可选：环境贴图 HDR 路径（用于初始化 envlight，空字符串表示不使用）")
    parser.add_argument("--environment_scale", type=float, default=1.0, help="环境贴图强度缩放")

    # 使用 get_combined_args 以便从 model_path/cfg_args 读取配置
    args = get_combined_args(parser)
    export_albedo_views(args)


if __name__ == "__main__":
    main()
