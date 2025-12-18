import argparse
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
import sys
import torch
from mmdet.apis import init_detector

# Reuse existing logic/constants from training code
from train_func import DETECTOR_PATHS, evaluate_from_saved_images
from utils.main_utils import coco_classes


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate saved final images with multiple detectors (quick eval).")
	parser.add_argument(
		"--exp_dir",
		type=str,
		default='/workspace/RGA/RGA_output/1124_204410_Beijing',
		help="训练输出的时间戳目录路径，例如：/workspace/RGA/RGA_output/1124_204410_Beijing"
	)
	parser.add_argument(
		"--anno_dir",
		type=str,
		default=None,
		help="标注目录（包含 LabelMe JSON 的 annos 文件夹）。若未提供，将尝试从 exp_dir/args.txt 中推断 source_path/annos。"
	)
	parser.add_argument(
		"--mmdet_base",
		type=str,
		default="/workspace/RGA/mmdet_files",
		help="MMDetection 配置与权重所在的基准目录。"
	)
	parser.add_argument(
		"--device",
		type=str,
		default=("cuda:0" if torch.cuda.is_available() else "cpu"),
		help="推理设备，如 cuda:0 或 cpu"
	)
	parser.add_argument(
		"--detectors",
		type=str,
		nargs="*",
		default=None,
		help="要评估的检测器名称列表（默认全部）。可选：%s" % ", ".join(DETECTOR_PATHS.keys())
	)
	parser.add_argument(
		"--target_class_name",
		type=str,
		default="car",
		help="目标类名称（用于 ASR 判定与 AP 计算），默认 car"
	)
	parser.add_argument(
		"--score_thresh",
		type=float,
		default=0.5,
		help="置信度阈值（与 IoU 联合用于攻击成功判定）"
	)
	parser.add_argument(
		"--log_name",
		type=str,
		default="quick_eval_log.txt",
		help="评估日志输出文件名（保存在 exp_dir 下）"
	)
	return parser.parse_args()


def try_infer_anno_dir(exp_dir: Path) -> Path | None:
	"""
	Try to infer dataset annotation dir from args.txt saved in the experiment directory.
	Look for a line like 'source_path: /path/to/dataset', and use '{source_path}/annos'.
	"""
	args_txt = exp_dir / "args.txt"
	if not args_txt.is_file():
		return None
	source_path = None
	try:
		with open(args_txt, "r", encoding="utf-8") as f:
			for line in f:
				if ":" not in line:
					continue
				k, v = line.split(":", 1)
				k = k.strip().lower()
				v = v.strip()
				if k in ["source_path", "dataset_source_path", "data_root", "data_dir"]:
					candidate = Path(v)
					if candidate.is_dir():
						source_path = candidate
						break
	except Exception:
		return None
	if source_path is None:
		return None
	anno_dir = source_path / "annos"
	return anno_dir if anno_dir.is_dir() else None


def evaluate_dir_with_detector(detector, image_dir: Path, anno_dir: Path, target_class_name: str, score_thresh: float):
	# Build a minimal args namespace for evaluate_from_saved_images
	eval_args = SimpleNamespace(target_class_name=target_class_name, score_thresh=score_thresh)
	return evaluate_from_saved_images(detector, image_dir, anno_dir, eval_args)


def main():
	args = parse_args()
	exp_dir = Path(args.exp_dir)
	assert exp_dir.is_dir(), f"Experiment directory not found: {exp_dir}"

	final_test_dir = exp_dir / "final_test_images"
	final_full_dir = exp_dir / "final_full_images"
	if not final_test_dir.is_dir() and not final_full_dir.is_dir():
		print(f"[错误] 未找到 {final_test_dir} 或 {final_full_dir}，请确认这是一个包含最终渲染图片的训练输出目录。")
		sys.exit(1)

	anno_dir = Path(args.anno_dir) if args.anno_dir is not None else try_infer_anno_dir(exp_dir)
	if anno_dir is None or not anno_dir.is_dir():
		print("[错误] 无法定位标注目录 annos。请通过 --anno_dir 显式指定，或确保 exp_dir/args.txt 中包含 source_path 以便自动推断。")
		sys.exit(2)

	detector_names = args.detectors if args.detectors else list(DETECTOR_PATHS.keys())
	for name in detector_names:
		if name not in DETECTOR_PATHS:
			print(f"[警告] 未知检测器名称: {name}，将跳过。可选值：{list(DETECTOR_PATHS.keys())}")
	detector_names = [n for n in detector_names if n in DETECTOR_PATHS]
	if not detector_names:
		print("[错误] 没有可评估的检测器。")
		sys.exit(3)

	mmdet_base = Path(args.mmdet_base)
	assert mmdet_base.is_dir(), f"mmdet_base 不存在: {mmdet_base}"

	log_path = exp_dir / args.log_name
	beijing_tz = timezone(timedelta(hours=8))
	with open(log_path, "a", encoding="utf-8") as f:
		f.write("\n\n==================================================\n")
		f.write(f"=== Quick Cross-Detector Evaluation ===\n")
		f.write("==================================================\n")
		f.write(f"Timestamp: {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}\n")
		f.write(f"Experiment Dir: {exp_dir}\n")
		f.write(f"Anno Dir: {anno_dir}\n")
		f.write(f"Target Class: {args.target_class_name}, Score Thresh: {args.score_thresh}\n\n")

	for det_name in detector_names:
		print(f"\n>>> [评估] 正在加载检测器: {det_name} ...")
		cfg_rel = DETECTOR_PATHS[det_name]['config']
		ckpt_rel = DETECTOR_PATHS[det_name]['ckpt']
		cfg_path = mmdet_base / cfg_rel
		ckpt_path = mmdet_base / ckpt_rel
		assert cfg_path.is_file(), f"Config file not found: {cfg_path}"
		assert ckpt_path.is_file(), f"Checkpoint file not found: {ckpt_path}"

		detector = init_detector(str(cfg_path), str(ckpt_path), device=args.device)
		if not hasattr(detector, 'CLASSES'):
			detector.CLASSES = coco_classes

		asr_test = succ_test = total_test = ap50_test = None
		asr_full = succ_full = total_full = ap50_full = None

		if final_test_dir.is_dir():
			asr_test, succ_test, total_test, ap50_test = evaluate_dir_with_detector(
				detector, final_test_dir, anno_dir, args.target_class_name, args.score_thresh
			)
			print(f"  - [Test Set] ASR: {asr_test:.4f} ({succ_test}/{total_test}), AP@0.5: {ap50_test:.4f}")
		else:
			print("  - [Test Set] 未找到 final_test_images，跳过。")

		if final_full_dir.is_dir():
			asr_full, succ_full, total_full, ap50_full = evaluate_dir_with_detector(
				detector, final_full_dir, anno_dir, args.target_class_name, args.score_thresh
			)
			print(f"  - [Full Set] ASR: {asr_full:.4f} ({succ_full}/{total_full}), AP@0.5: {ap50_full:.4f}")
		else:
			print("  - [Full Set] 未找到 final_full_images，跳过。")

		with open(log_path, "a", encoding="utf-8") as f:
			f.write(f"Detector: {det_name}\n")
			if asr_test is not None:
				f.write(f"  - [Test Set] ASR: {asr_test:.4f} ({succ_test}/{total_test}), AP@0.5: {ap50_test:.4f}\n")
			else:
				f.write("  - [Test Set] skipped\n")
			if asr_full is not None:
				f.write(f"  - [Full Set] ASR: {asr_full:.4f} ({succ_full}/{total_full}), AP@0.5: {ap50_full:.4f}\n")
			else:
				f.write("  - [Full Set] skipped\n")
			f.write("-" * 30 + "\n")

		# cleanup to free VRAM
		del detector
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	print(f"\n[消息] 评估完成。结果已保存至: {log_path}")


if __name__ == "__main__":
	main()


