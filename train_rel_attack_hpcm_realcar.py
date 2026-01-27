from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import csv
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import subprocess
import torch
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from envlight.utils import cubemap_to_latlong
from utils.main_utils import coco_classes
from mmdet.apis import init_detector
from lbm_relit import LBMRelighter
from utils.log_utils import TrainingLogger
# from attack_options_hpcm import get_attack_args
from attack_options_hpcm_realcar import get_attack_args

from submodules.envlight.envlight.light import EnvLight as EnvLightClass

from train_func_hdr import (
    DETECTOR_PATHS,
    save_visualization_grid,
    compute_batch_loss,
    precompute_lbm_disk_cache,
    evaluate,
    render_and_save_final_images_mw,
    render_and_save_final_images_ori,
    evaluate_from_saved_images,
    visualize_hdr_bank_from_dir,
    visualize_hdr_bases_with_random_views,
    load_hdr_image_and_tonemap,
    latest_iteration_ply,
    latest_checkpoint_pth,
    iteration_ply,
    first_existing,
)

# torch.autograd.set_detect_anomaly(True)

def compute_sh9_rgb_from_latlong(env_img: np.ndarray) -> np.ndarray:
    """
    Computes 2nd order (9 bands) spherical harmonics coefficients for each of the R,G,B channels,
    resulting in a 27-dimensional vector. This function is ported from the data preparation script
    to ensure consistency between training and inference.
    Assumes the input is a lat-long projection where rows correspond to theta in [0, pi]
    and columns to phi in [0, 2*pi).
    """
    assert env_img.ndim == 3 and env_img.shape[-1] == 3, "env_img must be HxWx3"
    H, W, _ = env_img.shape
    # Grid setup
    theta = np.linspace(0.0, np.pi, H, endpoint=False) + (np.pi / H) * 0.5
    phi = np.linspace(0.0, 2 * np.pi, W, endpoint=False) + (2 * np.pi / W) * 0.5
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")  # HxW
    # Direction vectors
    sin_t = np.sin(theta_grid)
    cos_t = np.cos(theta_grid)
    cos_p = np.cos(phi_grid)
    sin_p = np.sin(phi_grid)
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    # 9 SH basis functions
    Y00 = 0.28209479177387814 * np.ones_like(x)
    Y1m1 = 0.4886025119029199 * y
    Y10 = 0.4886025119029199 * z
    Y11 = 0.4886025119029199 * x
    Y2m2 = 1.0925484305920792 * x * y
    Y2m1 = 1.0925484305920792 * y * z
    Y20 = 0.31539156525252005 * (3.0 * z * z - 1.0)
    Y21 = 1.0925484305920792 * x * z
    Y22 = 0.5462742152960396 * (x * x - y * y)
    bases = [Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22]
    # Solid angle weights
    dtheta = np.pi / H
    dphi = 2.0 * np.pi / W
    domega = sin_t * dtheta * dphi  # HxW
    # Integrate over each channel
    L = env_img.astype(np.float32)  # HxWx3
    coefs = []
    for c in range(3):
        Lc = L[..., c]
        for B in bases:
            coefs.append(np.sum(Lc * B * domega, dtype=np.float64))
    coefs = np.array(coefs, dtype=np.float32)  # 27
    return coefs


def base_cubemap_to_sh(base_cubemap: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Converts an envlight base cubemap tensor into a Spherical Harmonics tensor,
    aligning with the LBM model's training data pipeline.
    """
    if base_cubemap is None:
        return torch.zeros(27, dtype=torch.float32, device=device)

    # 1. Convert base cubemap to latlong. The input `base_cubemap` is already a tensor.
    # The 'res' argument is required by this version of cubemap_to_latlong.
    latlong_t = cubemap_to_latlong(base_cubemap.to(device), res=(64, 128))
    latlong_np = latlong_t.detach().cpu().numpy().astype(np.float32)

    # 2. Resize latlong to fixed small resolution (128, 64)
    target_w, target_h = 128, 64
    # 修正：使用百分位数进行归一化，以抵抗离群值
    # max_val = np.percentile(latlong_np, 99.5) if np.max(latlong_np) > 0 else 1.0
    max_val = np.max(latlong_np) if np.max(latlong_np) > 0 else 1.0
    latlong_norm = np.clip(latlong_np / max_val, 0.0, 1.0)
    latlong_u8 = (latlong_norm * 255.0).astype(np.uint8)
    im = Image.fromarray(latlong_u8)
    im_resized = im.resize((target_w, target_h), resample=Image.BILINEAR)
    resized_u8 = np.asarray(im_resized).astype(np.float32) / 255.0
    latlong_resized = resized_u8 * max_val

    # 3. Compute SH coefficients
    sh_coeffs_np = compute_sh9_rgb_from_latlong(latlong_resized)

    # 4. Convert to tensor and return
    return torch.from_numpy(sh_coeffs_np).to(device)


def _parse_physical_from_name(name: str) -> tuple[int, int, int] | None:
	"""
	Parse discrete physical config from image/camera name.
	Expected patterns like:
	- ori_pitch20_angle80_distance5_sunny
	- xxx_pitch10_angle120_distance15
	Returns (pitch, angle, distance) as ints if all found, else None.
	"""
	if not isinstance(name, str) or len(name) == 0:
		return None
	# Accept multiple aliases for robustness
	pitch_m = re.search(r"(?:^|[_\-])pitch(?P<p>-?\d+)", name, flags=re.IGNORECASE)
	angle_m = re.search(r"(?:^|[_\-])(?:angle|azimuth|azi)(?P<a>-?\d+)", name, flags=re.IGNORECASE)
	dist_m = re.search(r"(?:^|[_\-])distance(?P<d>-?\d+)", name, flags=re.IGNORECASE)
	if pitch_m and angle_m and dist_m:
		try:
			return int(pitch_m.group("p")), int(angle_m.group("a")), int(dist_m.group("d"))
		except Exception:
			return None
	return None


def _edges_from_discrete_values(values: list[int]) -> np.ndarray:
	"""
	Create bin edges from discrete integer values for metadata / optional binning.
	For values [v0 < v1 < ...], edges are midpoints, with +/-0.5 padding at ends.
	"""
	vs = sorted(set(int(v) for v in values))
	if len(vs) == 0:
		return np.array([0.0, 1.0], dtype=np.float32)
	if len(vs) == 1:
		v = float(vs[0])
		return np.array([v - 0.5, v + 0.5], dtype=np.float32)
	mids = [(vs[i] + vs[i + 1]) * 0.5 for i in range(len(vs) - 1)]
	edges = [float(vs[0]) - 0.5] + [float(m) for m in mids] + [float(vs[-1]) + 0.5]
	return np.array(edges, dtype=np.float32)


class HPCMTable:
	"""
	Hard Physical Configuration Mining:
	- Discrete state: (pitch_bin, azimuth_bin, distance_bin, hdr_id)
	- Difficulty score: EMA of observed loss (no gradients)
	- Sampling: softmax(score / temperature)
	"""

	def __init__(
		self,
		state_bins: list[tuple[int, int, int]],
		hdr_names: list[str],
		temperature: float = 1.0,
		momentum: float = 0.9,
		init_score: float = 0.0,
		uniform_prob: float = 0.0,
	):
		self.state_bins = state_bins
		self.hdr_names = hdr_names
		self.temperature = float(max(1e-8, temperature))
		self.momentum = float(np.clip(momentum, 0.0, 0.999999))
		self.uniform_prob = float(np.clip(uniform_prob, 0.0, 1.0))
		self.num_states = int(len(state_bins))
		self.num_hdr = int(len(hdr_names))
		if self.num_states <= 0 or self.num_hdr <= 0:
			raise ValueError(f"HPCMTable requires num_states>0 and num_hdr>0, got {self.num_states}, {self.num_hdr}")
		self.scores = np.full((self.num_states * self.num_hdr,), float(init_score), dtype=np.float32)
		# Track how often each (state,hdr) config was updated; helps interpret table evolution.
		self.counts = np.zeros((self.num_states * self.num_hdr,), dtype=np.int32)

	def config_id(self, state_id: int, hdr_id: int) -> int:
		return int(state_id) * self.num_hdr + int(hdr_id)

	def decode(self, config_id: int) -> tuple[int, int]:
		config_id = int(config_id)
		state_id = config_id // self.num_hdr
		hdr_id = config_id % self.num_hdr
		return int(state_id), int(hdr_id)

	def sample_config(self, rng: np.random.Generator) -> int:
		n = self.scores.shape[0]
		if n == 1:
			return 0
		if self.uniform_prob > 0.0 and rng.random() < self.uniform_prob:
			return int(rng.integers(0, n))
		# softmax(scores / T)
		logits = self.scores.astype(np.float64) / float(self.temperature)
		logits -= np.max(logits)
		probs = np.exp(logits)
		s = float(np.sum(probs))
		if not np.isfinite(s) or s <= 0:
			return int(rng.integers(0, n))
		probs /= s
		return int(rng.choice(n, p=probs))

	def update(self, config_id: int, loss_value: float) -> None:
		config_id = int(config_id)
		loss_value = float(loss_value)
		old = float(self.scores[config_id])
		m = self.momentum
		self.scores[config_id] = np.float32(m * old + (1.0 - m) * loss_value)
		try:
			self.counts[config_id] = np.int32(int(self.counts[config_id]) + 1)
		except Exception:
			pass

	def save_npz(self, path: Path, pitch_edges: np.ndarray, az_edges: np.ndarray, dist_edges: np.ndarray) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		state_pitch = np.array([p for (p, a, d) in self.state_bins], dtype=np.int32)
		state_az = np.array([a for (p, a, d) in self.state_bins], dtype=np.int32)
		state_dist = np.array([d for (p, a, d) in self.state_bins], dtype=np.int32)
		np.savez_compressed(
			str(path),
			scores=self.scores.astype(np.float32),
			counts=self.counts.astype(np.int32),
			hdr_names=np.array(self.hdr_names, dtype=object),
			state_pitch=state_pitch,
			state_azimuth=state_az,
			state_distance=state_dist,
			pitch_edges=pitch_edges.astype(np.float32),
			azimuth_edges=az_edges.astype(np.float32),
			distance_edges=dist_edges.astype(np.float32),
			temperature=np.array([self.temperature], dtype=np.float32),
			momentum=np.array([self.momentum], dtype=np.float32),
			uniform_prob=np.array([self.uniform_prob], dtype=np.float32),
		)


def _safe_percentile(x: np.ndarray, q: float) -> float:
	try:
		return float(np.percentile(x, q))
	except Exception:
		return float("nan")


def _softmax_np(logits: np.ndarray) -> np.ndarray:
	logits = logits.astype(np.float64)
	logits = logits - np.max(logits)
	p = np.exp(logits)
	s = float(np.sum(p))
	if not np.isfinite(s) or s <= 0:
		return np.full_like(p, 1.0 / float(p.size), dtype=np.float64)
	return (p / s).astype(np.float64)


def _entropy(p: np.ndarray) -> float:
	p = np.asarray(p, dtype=np.float64)
	p = np.clip(p, 1e-12, 1.0)
	return float(-np.sum(p * np.log(p)))


def export_hpcm_monitor(
	*,
	save_dir: Path,
	step: int,
	hpcm: HPCMTable,
	pitch_vals: list[int],
	angle_vals: list[int],
	dist_vals: list[int],
	args: argparse.Namespace,
	prev_scores: np.ndarray | None,
	stats_history: list[dict],
	pitch_edges: np.ndarray,
	az_edges: np.ndarray,
	dist_edges: np.ndarray,
) -> np.ndarray | None:
	"""
	Export human-friendly artifacts for HPCM:
	- hpcm_monitor/hpcm_summary_latest.txt (+ per-step copies)
	- hpcm_monitor/hpcm_stats.csv (append per save)
	- hpcm_monitor/hpcm_topk_step_XXXXXX.csv
	- hpcm_monitor/plots/* (optional)
	- optional step-suffixed npz snapshots
	Returns a snapshot of current scores for delta computation on next save.
	"""
	monitor_dir = save_dir / "hpcm_monitor"
	(monitor_dir / "plots").mkdir(parents=True, exist_ok=True)
	(monitor_dir / "summaries").mkdir(parents=True, exist_ok=True)

	scores = np.asarray(hpcm.scores, dtype=np.float32)
	counts = np.asarray(getattr(hpcm, "counts", np.zeros_like(scores, dtype=np.int32)), dtype=np.int32)
	num_states = int(hpcm.num_states)
	num_hdr = int(hpcm.num_hdr)
	scores_2d = scores.reshape(num_states, num_hdr)
	counts_2d = counts.reshape(num_states, num_hdr)

	# Overall stats
	mean_score = float(np.mean(scores))
	std_score = float(np.std(scores))
	min_score = float(np.min(scores))
	max_score = float(np.max(scores))
	q50 = _safe_percentile(scores, 50.0)
	q90 = _safe_percentile(scores, 90.0)
	q99 = _safe_percentile(scores, 99.0)
	visited = int(np.sum(counts > 0))
	visited_frac = float(visited / max(1, counts.size))

	# Sampling distribution entropy (how peaky the miner is)
	try:
		probs = _softmax_np(scores.astype(np.float64) / float(hpcm.temperature))
		ent = _entropy(probs)
	except Exception:
		ent = float("nan")

	# Delta from previous snapshot
	changed = 0
	mean_abs_delta = float("nan")
	max_abs_delta = float("nan")
	if prev_scores is not None and isinstance(prev_scores, np.ndarray) and prev_scores.shape == scores.shape:
		delta = scores.astype(np.float64) - prev_scores.astype(np.float64)
		absd = np.abs(delta)
		changed = int(np.sum(absd > 1e-12))
		mean_abs_delta = float(np.mean(absd))
		max_abs_delta = float(np.max(absd))

	# Top-K configs (state,hdr) by score
	topk = int(max(1, getattr(args, "hpcm_topk", 20)))
	topk = int(min(topk, scores.size))
	top_idx = np.argsort(-scores.astype(np.float64))[:topk]

	# Also: top-K states by mean score across HDRs
	state_mean = scores_2d.mean(axis=1).astype(np.float64)
	top_state_idx = np.argsort(-state_mean)[: min(topk, state_mean.size)]

	def _decode_config(config_id: int) -> tuple[int, int, int, int, str]:
		sid, hid = hpcm.decode(int(config_id))
		p, a, d = hpcm.state_bins[sid]
		hname = hpcm.hdr_names[hid] if 0 <= hid < len(hpcm.hdr_names) else str(hid)
		# In camera-based mode, p is camera_idx, a and d are dummy (0)
		# In physical mode, p/a/d are pitch/angle/distance
		return int(p), int(a), int(d), int(hid), str(hname)

	# Write summary txt
	summary_lines: list[str] = []
	summary_lines.append(f"step: {int(step)}")
	summary_lines.append(f"num_states: {num_states}  num_hdr: {num_hdr}  num_configs: {scores.size}")
	summary_lines.append(f"temperature: {float(hpcm.temperature):.6g}  momentum: {float(hpcm.momentum):.6g}  uniform_prob: {float(hpcm.uniform_prob):.6g}")
	summary_lines.append(f"score: mean={mean_score:.6f} std={std_score:.6f} min={min_score:.6f} p50={q50:.6f} p90={q90:.6f} p99={q99:.6f} max={max_score:.6f}")
	summary_lines.append(f"visited_configs: {visited}/{counts.size} ({visited_frac*100.0:.2f}%)")
	summary_lines.append(f"sampling_entropy: {ent:.6f}")
	if prev_scores is not None:
		summary_lines.append(f"delta_vs_prev: changed={changed}/{scores.size} mean_abs_delta={mean_abs_delta:.6g} max_abs_delta={max_abs_delta:.6g}")
	summary_lines.append("")
	summary_lines.append(f"Top-{topk} configs by score (pitch, angle, distance, hdr, score, count):")
	for rank, cid in enumerate(top_idx, start=1):
		p, a, d, hid, hname = _decode_config(int(cid))
		sv = float(scores[int(cid)])
		cv = int(counts[int(cid)])
		summary_lines.append(f"{rank:02d}. pitch={p:>4d} angle={a:>4d} distance={d:>4d} hdr={hid:>3d}({hname}) score={sv:.6f} count={cv}")
	summary_lines.append("")
	summary_lines.append(f"Top-{min(topk, state_mean.size)} states by mean(score over HDR) (pitch, angle, distance, mean_score):")
	for rank, sid in enumerate(top_state_idx, start=1):
		p, a, d = hpcm.state_bins[int(sid)]
		summary_lines.append(f"{rank:02d}. pitch={int(p):>4d} angle={int(a):>4d} distance={int(d):>4d} mean_score={float(state_mean[int(sid)]):.6f}")
	summary_lines.append("")
	summary_lines.append("npz_fields:")
	summary_lines.append("  - scores: float32 [num_states*num_hdr]")
	summary_lines.append("  - counts: int32   [num_states*num_hdr]  (update frequency)")
	summary_lines.append("  - hdr_names, state_pitch/state_azimuth/state_distance, edges, temperature/momentum/uniform_prob")

	latest_summary = monitor_dir / "hpcm_summary_latest.txt"
	step_summary = monitor_dir / "summaries" / f"hpcm_summary_step_{int(step):06d}.txt"
	latest_summary.write_text("\n".join(summary_lines), encoding="utf-8")
	step_summary.write_text("\n".join(summary_lines), encoding="utf-8")

	# Append stats csv
	stats_csv = monitor_dir / "hpcm_stats.csv"
	row = {
		"step": int(step),
		"mean": mean_score,
		"std": std_score,
		"min": min_score,
		"p50": q50,
		"p90": q90,
		"p99": q99,
		"max": max_score,
		"visited": visited,
		"visited_frac": visited_frac,
		"entropy": ent,
		"changed": int(changed),
		"mean_abs_delta": float(mean_abs_delta),
		"max_abs_delta": float(max_abs_delta),
	}
	stats_history.append(row)
	write_header = not stats_csv.exists()
	with stats_csv.open("a", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(row.keys()))
		if write_header:
			writer.writeheader()
		writer.writerow(row)

	# Write TopK CSV
	topk_csv = monitor_dir / f"hpcm_topk_step_{int(step):06d}.csv"
	with topk_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["rank", "pitch", "angle", "distance", "hdr_id", "hdr_name", "score", "count"])
		for rank, cid in enumerate(top_idx, start=1):
			p, a, d, hid, hname = _decode_config(int(cid))
			writer.writerow([rank, p, a, d, hid, hname, float(scores[int(cid)]), int(counts[int(cid)])])

	# Optionally export full CSV (can be large)
	if bool(getattr(args, "hpcm_export_full_csv", False)):
		full_csv = monitor_dir / f"hpcm_full_step_{int(step):06d}.csv"
		with full_csv.open("w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["config_id", "state_id", "hdr_id", "pitch", "angle", "distance", "hdr_name", "score", "count"])
			for cid in range(scores.size):
				sid, hid = hpcm.decode(int(cid))
				p, a, d = hpcm.state_bins[int(sid)]
				hname = hpcm.hdr_names[int(hid)] if 0 <= int(hid) < len(hpcm.hdr_names) else str(hid)
				writer.writerow([cid, sid, hid, int(p), int(a), int(d), hname, float(scores[int(cid)]), int(counts[int(cid)])])

	# Optional plots
	if bool(getattr(args, "hpcm_export_plots", True)):
		try:
			import matplotlib.pyplot as plt

			plots_dir = monitor_dir / "plots"

			# 1) Histogram of scores
			fig = plt.figure(figsize=(8, 4.5))
			plt.hist(scores.astype(np.float64), bins=60)
			plt.title(f"HPCM score distribution (step {int(step)})")
			plt.xlabel("score")
			plt.ylabel("count")
			plt.grid(True, alpha=0.3)
			fig.tight_layout()
			fig.savefig(plots_dir / f"hpcm_score_hist_step_{int(step):06d}.png", dpi=160)
			plt.close(fig)

			# 2) Stats curves (over saves)
			if len(stats_history) >= 2:
				steps = [int(r["step"]) for r in stats_history]
				means = [float(r["mean"]) for r in stats_history]
				p90s = [float(r["p90"]) for r in stats_history]
				maxs = [float(r["max"]) for r in stats_history]
				fig = plt.figure(figsize=(9, 4.8))
				plt.plot(steps, means, label="mean")
				plt.plot(steps, p90s, label="p90")
				plt.plot(steps, maxs, label="max")
				plt.title("HPCM score stats over time (per save)")
				plt.xlabel("step")
				plt.ylabel("score")
				plt.grid(True, alpha=0.3)
				plt.legend()
				fig.tight_layout()
				fig.savefig(plots_dir / "hpcm_score_stats_curve.png", dpi=160)
				plt.close(fig)

			# 3) Pitch-slice heatmaps for mean(score over HDR) and visit counts
			max_pitches = int(getattr(args, "hpcm_plot_max_pitches", 12))
			if max_pitches != 0:
				P = int(len(pitch_vals))
				A = int(len(angle_vals))
				D = int(len(dist_vals))
				if P > 0 and A > 0 and D > 0 and (P * A * D) == num_states:
					state_mean_3d = state_mean.astype(np.float64).reshape(P, A, D)
					state_cnt_3d = counts_2d.sum(axis=1).astype(np.int64).reshape(P, A, D)

					# Choose pitches to plot: evenly spaced if too many.
					if max_pitches > 0 and P > max_pitches:
						idxs = np.linspace(0, P - 1, max_pitches).round().astype(int).tolist()
						pitch_idxs = sorted(set(int(i) for i in idxs))
					else:
						pitch_idxs = list(range(P))

					# Create per-save subdir for heatmaps to avoid overwriting
					hm_dir = plots_dir / f"heatmaps_step_{int(step):06d}"
					hm_dir.mkdir(parents=True, exist_ok=True)

					for ip in pitch_idxs:
						p = int(pitch_vals[ip])
						hm = state_mean_3d[ip]
						fig, ax = plt.subplots(figsize=(7.2, 4.8))
						im = ax.imshow(hm, aspect="auto")
						ax.set_title(f"mean(score over HDR) | pitch={p} | step={int(step)}")
						ax.set_xlabel("distance")
						ax.set_ylabel("angle")
						ax.set_xticks(list(range(D)))
						ax.set_xticklabels([str(int(v)) for v in dist_vals], rotation=45, ha="right", fontsize=8)
						ax.set_yticks(list(range(A)))
						ax.set_yticklabels([str(int(v)) for v in angle_vals], fontsize=8)
						fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
						fig.tight_layout()
						fig.savefig(hm_dir / f"hpcm_mean_heatmap_pitch{p}_step_{int(step):06d}.png", dpi=160)
						plt.close(fig)

						hc = state_cnt_3d[ip].astype(np.float64)
						fig, ax = plt.subplots(figsize=(7.2, 4.8))
						im = ax.imshow(hc, aspect="auto")
						ax.set_title(f"visit_count(sum over HDR) | pitch={p} | step={int(step)}")
						ax.set_xlabel("distance")
						ax.set_ylabel("angle")
						ax.set_xticks(list(range(D)))
						ax.set_xticklabels([str(int(v)) for v in dist_vals], rotation=45, ha="right", fontsize=8)
						ax.set_yticks(list(range(A)))
						ax.set_yticklabels([str(int(v)) for v in angle_vals], fontsize=8)
						fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
						fig.tight_layout()
						fig.savefig(hm_dir / f"hpcm_count_heatmap_pitch{p}_step_{int(step):06d}.png", dpi=160)
						plt.close(fig)
		except Exception as e:
			# Avoid crashing training due to visualization failures.
			print(f"[警告] [HPCM] 导出监控图失败: {e}")

	return scores.copy()

def main():
	# =================================================================================
	# 1. 参数解析（HPCM: MIN-only, step-based）
	# =================================================================================
	args, model_params, pipeline_params = get_attack_args()
	# HPCM 不使用基于梯度的内部最大化（SGLD/PGD）、不使用 min-max、也不使用 replay buffer。
	initial_env_base_cpu = None

	# =================================================================================
	# 2. 环境与路径设置
	# =================================================================================
	device = torch.device(args.device)
	save_dir = Path(args.save_dir)
	# Resolve relative save_dir to repository directory to avoid cwd permission issues
	if not save_dir.is_absolute():
		repo_dir = Path(__file__).resolve().parent
		save_dir = repo_dir / save_dir

	# New: Create a timestamped subdirectory for all outputs using Beijing Time (UTC+8)
	beijing_tz = timezone(timedelta(hours=8))
	timestamp = datetime.now(beijing_tz).strftime("%m%d_%H%M%S") + "_Beijing"
	save_dir = save_dir / timestamp

	# If a file exists with the same name, redirect to a new folder name
	if save_dir.exists() and not save_dir.is_dir():
		save_dir = save_dir.with_name(save_dir.name + "_dir")
	# Try mkdir with graceful fallbacks
	try:
		save_dir.mkdir(parents=True, exist_ok=True)
	except PermissionError:
		fallback1 = Path('/workspace/RGA') / save_dir.name
		try:
			fallback1.mkdir(parents=True, exist_ok=True)
			save_dir = fallback1
		except PermissionError as e:
			from time import time as _now
			fallback2 = Path('/workspace/RGA') / f"{save_dir.name}_{int(_now())}"
			try:
				fallback2.mkdir(parents=True, exist_ok=True)
				save_dir = fallback2
			except PermissionError as e2:
				raise PermissionError(
					f"Failed to create save dir at '{save_dir}' and '{fallback1}'. "
					"Please set --save_dir to a writable path under /workspace."
				) from e2
	print(f"[消息] 输出将保存到: {save_dir}")

	# Save command line arguments to a file
	args_save_path = save_dir / "args.txt"
	with open(args_save_path, 'w') as f:
		for k, v in sorted(vars(args).items()):
			f.write(f"{k}: {v}\n")
	print(f"[消息] 命令行参数已保存到: {args_save_path}")

	# =================================================================================
	# 3. 加载场景与高斯模型
	# =================================================================================
	dataset = model_params.extract(args)
	# dataset.eval = True # Reverted: We will manually split cameras instead.

	if args.environment_texture != "":
		gaussians = GaussianModel(dataset.sh_degree, environment_texture=args.environment_texture, environment_scale=args.environment_scale)
	else:
		gaussians = GaussianModel(dataset.sh_degree)

	if args.environment_texture == "":
		print("[消息] 未提供 environment_texture，将尝试从最新的检查点加载 'envlight'...")
		model_dir = Path(dataset.model_path)
		latest_ckpt_path = latest_checkpoint_pth(model_dir)
		if latest_ckpt_path:
			print(f"[消息] 找到最新检查点: {latest_ckpt_path}")
			try:
				# The .pth file contains a tuple: (captured_model_tuple, iteration_number)
				ckpt_data_tuple, _ = torch.load(str(latest_ckpt_path), map_location=device)

				# Find the envlight state_dict by searching for a dictionary that is NOT the optimizer state.
				# This is more robust than assuming a fixed index.
				envlight_state_dict = None
				for item in ckpt_data_tuple:
					if isinstance(item, dict):
						# A simple heuristic to distinguish optimizer state_dict from others.
						# Optimizer state has 'state' and 'param_groups' keys.
						if 'state' in item and 'param_groups' in item:
							continue
						else:
							envlight_state_dict = item
							break # Found it

				if envlight_state_dict is not None:
					print("[消息] 正在从检查点加载 'env_light'...")
					# Load the state into the default envlight object that comes with the gaussians model
					gaussians.envlight.load_state_dict(envlight_state_dict)
					print("[消息] 成功加载 'envlight'。")
					# 加载后重建 base 与 mips，确保采样链与参数一致
					try:
						gaussians.envlight.build_base()
						gaussians.envlight.build_mips()
						print("[消息] 已根据加载参数重建 envlight base 与 mips。")
						# 保存初始 base（CPU 拷贝）
						try:
							initial_env_base_cpu = gaussians.envlight.base.detach().cpu().clone()
						except Exception:
							initial_env_base_cpu = None
					except Exception as e:
						print(f"[警告] 重建 envlight base/mips 失败: {e}")

				else:
					print("[消息] 在检查点中未找到 'env_light' 的 state_dict，将在无环境光情况下继续。")
			except Exception as e:
				print(f"[消息] 从检查点加载 envlight 失败: {e}")
		else:
			print("[消息] 未找到 '.pth' 检查点文件，将在无环境光情况下继续。")

	# 若未能从 checkpoint 设置初始 base，则以当前 base 为初始备份
	if initial_env_base_cpu is None:
		try:
			initial_env_base_cpu = gaussians.envlight.base.detach().cpu().clone()
		except Exception:
			initial_env_base_cpu = None

	scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
	gaussians.get_diffuse_occ()

	# Manually clone the gaussians model once before training to avoid deepcopy issues
	gaussians_original = GaussianModel(dataset.sh_degree)
	gaussians_original.active_sh_degree = gaussians.active_sh_degree
	gaussians_original._xyz = torch.nn.Parameter(gaussians._xyz.clone().detach())
	gaussians_original._features_dc = torch.nn.Parameter(gaussians._features_dc.clone().detach())
	gaussians_original._features_rest = torch.nn.Parameter(gaussians._features_rest.clone().detach())
	gaussians_original._scaling = torch.nn.Parameter(gaussians._scaling.clone().detach())
	gaussians_original._rotation = torch.nn.Parameter(gaussians._rotation.clone().detach())
	gaussians_original._opacity = torch.nn.Parameter(gaussians._opacity.clone().detach())
	gaussians_original._albedo_init = torch.nn.Parameter(gaussians._albedo_init.clone().detach())
	if hasattr(gaussians, '_metallic_init'):
		gaussians_original._metallic_init = torch.nn.Parameter(gaussians._metallic_init.clone().detach())
	if hasattr(gaussians, '_roughness_init'):
		gaussians_original._roughness_init = torch.nn.Parameter(gaussians._roughness_init.clone().detach())
	gaussians_original.envlight.load_state_dict(gaussians.envlight.state_dict())
	# 同步一次 envlight.base，state_dict 不包含非参数的 base
	with torch.no_grad():
		try:
			gaussians_original.envlight.base = gaussians.envlight.base.detach().clone()
		except Exception:
			pass
	gaussians_original.max_radii2D = gaussians.max_radii2D.clone().detach()
	gaussians_original.diffuse_occ = gaussians.diffuse_occ.clone().detach()
	gaussians_original.diffuse_direction_samples = gaussians.diffuse_direction_samples.clone().detach()
	if hasattr(gaussians, 'min_pts') and gaussians.min_pts is not None:
		gaussians_original.min_pts = gaussians.min_pts.clone().detach()
	if hasattr(gaussians, 'max_pts') and gaussians.max_pts is not None:
		gaussians_original.max_pts = gaussians.max_pts.clone().detach()
	gaussians_original.get_diffuse_occ()
	
	with torch.no_grad():
		initial_raw_albedo = gaussians._albedo_init.data
		print(f"[消息] [检查] 模型加载后, _albedo_init 范围: [{initial_raw_albedo.min().item():.4f}, {initial_raw_albedo.max().item():.4f}]")
		gaussians._albedo_init.data.zero_()
		zeroed_raw_albedo = gaussians._albedo_init.data
		print(f"[消息] [修正] _albedo_init 已全部设置为 0. 新范围: [{zeroed_raw_albedo.min().item():.4f}, {zeroed_raw_albedo.max().item():.4f}]")

	# --- Perturb Initial Albedo if Enabled ---
	if args.perturb_albedo:
		print("[消息] [扰动] 启用反照率随机初始化...")
		with torch.no_grad():
			albedo = gaussians.get_albedo_init
			min_val, max_val = albedo.min(), albedo.max()
			original_mean = albedo.mean().item()
			
			if args.albedo_init_method == 'perturb':
				print(f"[消息] [扰动] 使用 'perturb' 方法进行初始化...")
				budget = (max_val - min_val) * args.perturb_budget_factor
				# 生成初始化扰动 [-budget, budget] 并添加在原始albedo上
				perturbation = (torch.rand_like(albedo) * 2 - 1) * budget
				gaussians._albedo_init.data += perturbation
				# gaussians._albedo_init.data.zero_()

			elif args.albedo_init_method == 'random':
				print(f"[消息] [扰动] 使用 'random' 方法进行初始化...")
				# 在 [min_val, max_val] 范围内完全随机初始化
				random_values = min_val + (max_val - min_val) * torch.rand_like(albedo)
				gaussians._albedo_init.data = random_values

			perturbed_mean = gaussians.get_albedo_init.mean().item()

			print(f"[消息] [扰动] 原始反照率范围: [{min_val:.4f}, {max_val:.4f}]")
			print(f"[消息] [扰动] 扰动前平均反照率: {original_mean:.4f}")
			print(f"[消息] [扰动] 扰动后平均反照率: {perturbed_mean:.4f}")

	# DEBUG: print number of gaussians (ellipsoids)
	try:
		num_gaussians = int(gaussians.get_xyz.shape[0])
		print(f"[消息] [模型] 高斯数量={num_gaussians}")
		# opacity / albedo quick stats
		op = gaussians.get_opacity
		print(f"[消息] [模型] 不透明度统计: 最小={float(op.min().item()):.6f}, 最大={float(op.max().item()):.6f}, 平均={float(op.mean().item()):.6f}")
		alb = gaussians.get_albedo_init
		print(f"[消息] [模型] 反照率统计: 最小={float(alb.min().item()):.6f}, 最大={float(alb.max().item()):.6f}, 平均={float(alb.mean().item()):.6f}")
	except Exception as e:
		print(f"[消息] [模型] 读取高斯数量失败: {e}")

	# =================================================================================
	# 4. 初始化渲染器与优化器
	# =================================================================================
	bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
	bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
	pipe = pipeline_params.extract(args)

	# Dynamically create the optimizer based on user's choice
	if args.optimizer == 'adam':
		optimizer_min = torch.optim.Adam([gaussians._albedo_init], lr=args.lr)
		print(f"[消息] [优化器] 使用 Adam, 学习率: {args.lr}")
	elif args.optimizer == 'sgd':
		optimizer_min = torch.optim.SGD([gaussians._albedo_init], lr=args.lr, momentum=args.momentum)
		print(f"[消息] [优化器] 使用 SGD, 学习率: {args.lr}, 动量: {args.momentum}")
	elif args.optimizer == 'adamw':
		optimizer_min = torch.optim.AdamW([gaussians._albedo_init], lr=args.lr)
		print(f"[消息] [优化器] 使用 AdamW, 学习率: {args.lr}")
	else:
		# This case should not be reached due to 'choices' in argparse
		raise ValueError(f"未知的优化器类型: {args.optimizer}")

	# HPCM: 冻结 envlight（不做梯度最大化，只进行离散 HDR 切换）
	try:
		for p in gaussians.envlight.parameters():
			p.requires_grad = False
	except Exception:
		pass
	try:
		for p in gaussians_original.envlight.parameters():
			p.requires_grad = False
	except Exception:
		pass


	# =================================================================================
	# 5. 初始化目标检测器与损失函数
	# =================================================================================
	print("[消息] 正在初始化检测器...")

	base_path = Path('/workspace/RGA/mmdet_files')
	
	# detector_paths removed here, using global DETECTOR_PATHS

	selected_detector = DETECTOR_PATHS.get(args.detector)

	if selected_detector is None:
		raise ValueError(f"Detector '{args.detector}' not found in hardcoded paths. Available: {list(DETECTOR_PATHS.keys())}")

	cfg_path = base_path / selected_detector['config']
	ckpt_path = base_path / selected_detector['ckpt']
	print(f"[消息] 配置文件路径: {cfg_path}")
	print(f"[消息] 权重文件路径: {ckpt_path}")
	if not cfg_path.is_file():
		raise FileNotFoundError(f"Config file not found: {cfg_path}. Please check the path and ensure 'mmdet_files' is set up correctly.")
	if not ckpt_path.is_file():
		raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}. Please check the path and ensure 'mmdet_files' is set up correctly.")

	cfg_path, ckpt_path = str(cfg_path), str(ckpt_path)

	args.detector_cfg = cfg_path
	args.detector_ckpt = ckpt_path
	print(f"[消息] 配置文件路径: {cfg_path}")
	print(f"[消息] 权重文件路径: {ckpt_path}")
	# MMDetLoss is a wrapper, but for custom loss logic, we use the model directly.
	detector = init_detector(cfg_path, ckpt_path, device=device)
	for p in detector.parameters():
		p.requires_grad = False
	detector.eval()
	if not hasattr(detector, 'CLASSES'):
		detector.CLASSES = coco_classes
	print("[消息] 检测器初始化完成。")

	# =================================================================================
	# 5.5. (Optional) Initialize LBM Relighter
	# =================================================================================
	relighter = None
	if args.enable_lbm_relight:
		print("[消息] LBM 背景重打光已启用。正在初始化 LBMRelighter...")
		try:
			relighter = LBMRelighter(ckpt_dir=args.lbm_ckpt_dir, device=device)
			print("[消息] LBMRelighter 初始化完成。")
		except Exception as e:
			print(f"[警告] LBMRelighter 初始化失败: {e}。将禁用背景重打光。")
			relighter = None


	# =================================================================================
	# 6. HPCM：构建离散状态表并进行 step-based MIN 优化
	# =================================================================================
	render_global_step = int(getattr(args, "global_step_start", 60000))
	batch_size = int(args.batch_size)

	# New: Initialize TrainingLogger
	logger = TrainingLogger(save_dir)

	# --- Manual Train/Test Split (for final evaluation only) ---
	print("[消息] 正在抽取测试子集（训练集=全集，HPCM: step-based training）...")
	all_cameras = list(scene.getTrainCameras())  # With eval=False, this gets all cameras
	random.shuffle(all_cameras)
	if args.max_cams > 0:
		all_cameras = all_cameras[: args.max_cams]
	test_size = int(len(all_cameras) * 0.1)
	test_cameras = all_cameras[-test_size:] if test_size > 0 else []
	train_cameras_all = all_cameras  # Training uses full set (including test subset)

	# Filter training cameras by annotation existence to avoid empty batches
	anno_dir = Path(dataset.source_path) / "annos"
	train_cameras_anno = [c for c in train_cameras_all if (anno_dir / f"{c.image_name}.json").exists()]

	# HPCM: Try to parse physical parameters, but fallback to camera-based configs if parsing fails
	parsed_train = []
	use_physical_states = False
	for cam in train_cameras_anno:
		if _parse_physical_from_name(cam.image_name) is not None:
			parsed_train.append(cam)
			use_physical_states = True
	
	# If we can parse at least one camera, use physical state mode; otherwise use camera-based mode
	if use_physical_states:
		train_cameras = parsed_train
		print(
			f"[消息] 划分完成. 训练集(全集，有标注且可解析pitch/angle/distance): {len(train_cameras)} 张, "
			f"测试子集: {len(test_cameras)} 张"
		)
		if len(train_cameras) == 0:
			raise RuntimeError(
				"HPCM requires camera.image_name to contain pitch/angle/distance (e.g. '...pitch20_angle80_distance5...'), "
				f"but none were parsable. anno_dir={anno_dir}"
			)
	else:
		# Fallback: use all cameras with annotations, treat each camera as a separate state
		train_cameras = train_cameras_anno
		print(
			f"[消息] 划分完成. 训练集(全集，有标注，无法解析pitch/angle/distance，使用camera-based模式): {len(train_cameras)} 张, "
			f"测试子集: {len(test_cameras)} 张"
		)
		if len(train_cameras) == 0:
			raise RuntimeError(
				f"No cameras with annotations found. anno_dir={anno_dir}"
			)

	# --- Load discrete HDR bases (EnvMap bank) ---
	hdr_bases_cpu: list[torch.Tensor] = []
	hdr_names: list[str] = []
	# Ablation helper:
	# If LBM relighting is disabled, and we are doing HPCM sampling, freeze HDR for the VEHICLE to
	# the checkpoint envlight base (no hdr_bank_dir). This keeps only "background mixing" as the
	# changing factor when comparing enable_lbm_relight on/off with HPCM still enabled.
	freeze_hdr_to_ckpt = (not bool(getattr(args, "enable_lbm_relight", True))) and (
		str(getattr(args, "hpcm_sampling", "hpcm")).lower() == "hpcm"
	)
	if freeze_hdr_to_ckpt:
		if isinstance(getattr(args, "environment_texture", ""), str) and len(args.environment_texture) > 0:
			print(f"[消息] [HDR-Ablation] enable_lbm_relight=False 且 hpcm_sampling=hpcm：忽略 environment_texture={args.environment_texture}")
		if isinstance(getattr(args, "hdr_bank_dir", ""), str) and len(args.hdr_bank_dir) > 0:
			print(f"[消息] [HDR-Ablation] enable_lbm_relight=False 且 hpcm_sampling=hpcm：忽略 hdr_bank_dir={args.hdr_bank_dir}")
		base_cpu = None
		try:
			if initial_env_base_cpu is not None:
				base_cpu = initial_env_base_cpu.detach().cpu().clone()
		except Exception:
			base_cpu = None
		if base_cpu is None:
			try:
				base_cpu = gaussians.envlight.base.detach().cpu().clone()
			except Exception:
				base_cpu = None
		if base_cpu is None:
			raise RuntimeError(
				"[HDR-Ablation] Failed to get checkpoint envlight base. "
				"Please ensure a checkpoint with envlight exists, or pass --environment_texture."
			)
		hdr_bases_cpu = [base_cpu]
		hdr_names = ["ckpt_base"]
		print("[消息] [HDR-Ablation] 已固定车辆 HDR 为 checkpoint base（HPCM 的 HDR 维度=1）。")
	else:
		if isinstance(getattr(args, "environment_texture", ""), str) and len(args.environment_texture) > 0:
			print(f"[消息] [HDR] 使用单一 environment_texture: {args.environment_texture}")
			tmp_env = EnvLightClass(
				path=str(args.environment_texture),
				device=("cuda" if torch.cuda.is_available() else "cpu"),
				scale=args.environment_scale,
				min_res=16,
				max_res=512,
				trainable=False,
			)
			hdr_bases_cpu.append(tmp_env.base.detach().cpu().clone())
			hdr_names.append(Path(args.environment_texture).name)
			del tmp_env
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		else:
			hdr_bank_dir = getattr(args, "hdr_bank_dir", "")
			hdr_dir_path = Path(hdr_bank_dir) if isinstance(hdr_bank_dir, str) else None
			if hdr_dir_path is None or not hdr_dir_path.is_dir():
				print(f"[警告] [HDR] hdr_bank_dir 无效: {hdr_bank_dir}，将回退到 checkpoint base。")
			else:
				hdr_files = sorted([p for p in hdr_dir_path.iterdir() if p.suffix.lower() in [".hdr", ".exr"]])
				print(f"[消息] [HDR Bank] 在 '{hdr_bank_dir}' 发现 {len(hdr_files)} 个 HDR/EXR 文件，开始预加载 base...")
				for fp in hdr_files:
					try:
						tmp_env = EnvLightClass(
							path=str(fp),
							device=("cuda" if torch.cuda.is_available() else "cpu"),
							scale=args.environment_scale,
							min_res=16,
							max_res=512,
							trainable=False,
						)
						hdr_bases_cpu.append(tmp_env.base.detach().cpu().clone())
						hdr_names.append(fp.name)
						print(f"[消息] [HDR Bank] 已载入 base: {fp.name}")
					except Exception as e:
						print(f"[警告] [HDR Bank] 载入失败: {fp.name}: {e}")
					finally:
						try:
							del tmp_env
						except Exception:
							pass
						if torch.cuda.is_available():
							torch.cuda.empty_cache()

	# Fallback: use checkpoint base as a single envmap if none loaded
	if len(hdr_bases_cpu) == 0 and initial_env_base_cpu is not None:
		print("[消息] [HDR] 未加载任何 HDR 文件，使用 checkpoint 的 envlight.base 作为唯一 EnvMap。")
		hdr_bases_cpu = [initial_env_base_cpu.detach().cpu().clone()]
		hdr_names = ["ckpt_base"]

	if len(hdr_bases_cpu) == 0:
		raise RuntimeError("No HDR envmaps available (hdr_bank_dir empty/invalid and no checkpoint base).")
	
	# --- Option B: Precompute SH coefficients for each HDR base ---
	hdr_sh_cpu: list[torch.Tensor] | None = None
	if bool(getattr(args, "hpcm_precompute_hdr_sh", True)):
		print(f"[消息] [HPCM-Speed] 正在预计算 {len(hdr_bases_cpu)} 个 HDR 的 SH 系数...")
		hdr_sh_cpu = []
		for i, base_cpu in enumerate(hdr_bases_cpu):
			try:
				with torch.no_grad():
					# Use a temporary move to device for computation
					sh = base_cubemap_to_sh(base_cpu.to(device), device).detach().cpu()
				hdr_sh_cpu.append(sh)
			except Exception as e:
				print(f"[警告] [HPCM-Speed] 预计算失败 ({hdr_names[i]}): {e}")
				hdr_sh_cpu.append(torch.zeros(27, dtype=torch.float32))
		print("[消息] [HPCM-Speed] SH 预计算完成。")

	# --- Option C: Initialize Relight Cache ---
	relight_cache = {} if bool(getattr(args, "hpcm_enable_relight_cache", True)) else None
	
	# --- Option B: Precompute SH coefficients for each HDR base ---
	hdr_sh_cpu: list[torch.Tensor] | None = None
	if bool(getattr(args, "hpcm_precompute_hdr_sh", True)):
		print(f"[消息] [HPCM-Speed] 正在预计算 {len(hdr_bases_cpu)} 个 HDR 的 SH 系数...")
		hdr_sh_cpu = []
		for i, base_cpu in enumerate(hdr_bases_cpu):
			try:
				with torch.no_grad():
					sh = base_cubemap_to_sh(base_cpu.to(device), device).detach().cpu()
				hdr_sh_cpu.append(sh)
			except Exception as e:
				print(f"[警告] [HPCM-Speed] 预计算失败 ({hdr_names[i]}): {e}")
				hdr_sh_cpu.append(torch.zeros(27, dtype=torch.float32))
		print("[消息] [HPCM-Speed] SH 预计算完成。")

	# --- Option C: Initialize Relight Cache ---
	relight_cache = {} if bool(getattr(args, "hpcm_enable_relight_cache", True)) else None

	# --- Build discrete state bins from cameras ---
	# --- (Optional) Precompute LBM backgrounds to disk cache for cross-run reuse ---
	lbm_disk_cache_dir = getattr(args, "lbm_disk_cache_dir", "")
	if (
		isinstance(lbm_disk_cache_dir, str)
		and len(lbm_disk_cache_dir) > 0
		and bool(getattr(args, "lbm_disk_cache_precompute", False))
	):
		try:
			precompute_lbm_disk_cache(
				cameras=train_cameras,
				gaussians_original=gaussians_original,
				pipe=pipe,
				bg=bg,
				dataset=dataset,
				relighter=relighter,
				hdr_bases_cpu=hdr_bases_cpu,
				hdr_sh_cpu=hdr_sh_cpu,
				args=args,
				global_step=render_global_step,
				cache_dir=Path(lbm_disk_cache_dir),
				force_rebuild=bool(getattr(args, "lbm_disk_cache_force_rebuild", False)),
			)
		except Exception as e:
			print(f"[警告] [LBM-DiskCache] 预渲染失败: {e}。将继续正常训练并在训练中懒加载/懒写入缓存。")

	# Build state space based on whether we can parse physical parameters
	if use_physical_states:
		# Mode 1: Physical state space (pitch x angle x distance)
		parsed_triplets: list[tuple[int, int, int]] = []
		for cam in train_cameras:
			t = _parse_physical_from_name(cam.image_name)
			# should never be None due to filtering above, but keep it strict
			if t is None:
				raise RuntimeError(f"HPCM name parsing failed unexpectedly for camera: {cam.image_name}")
			parsed_triplets.append(t)

		# Unique discrete values from dataset
		pitch_vals = sorted({t[0] for t in parsed_triplets})
		angle_vals = sorted({t[1] for t in parsed_triplets})
		dist_vals = sorted({t[2] for t in parsed_triplets})

		# Metadata edges for saving (not used for sampling/bucketing)
		pitch_edges = _edges_from_discrete_values(pitch_vals)
		az_edges = _edges_from_discrete_values(angle_vals)
		dist_edges = _edges_from_discrete_values(dist_vals)

		# Camera lookup: (pitch,angle,distance) -> list of camera indices
		cam_map: dict[tuple[int, int, int], list[int]] = {}
		for i, t in enumerate(parsed_triplets):
			key = (int(t[0]), int(t[1]), int(t[2]))
			cam_map.setdefault(key, []).append(i)

		# HPCM global state space = pitch x angle x distance (cartesian product)
		state_bins: list[tuple[int, int, int]] = [
			(int(p), int(a), int(d))
			for p in pitch_vals
			for a in angle_vals
			for d in dist_vals
		]
		cams_by_state: list[list[int]] = [cam_map.get(st, []) for st in state_bins]
		state_id_by_triplet: dict[tuple[int, int, int], int] = {st: i for i, st in enumerate(state_bins)}
		
		print(f"[消息] [HPCM] 物理状态模式: 状态数(来自相机离散化): {len(state_bins)}；EnvMap 数: {len(hdr_bases_cpu)}")
	else:
		# Mode 2: Camera-based state space (each camera is a state)
		state_bins: list[tuple[int, int, int]] = [(i, 0, 0) for i in range(len(train_cameras))]  # Dummy values for compatibility
		cams_by_state: list[list[int]] = [[i] for i in range(len(train_cameras))]  # Each state has exactly one camera
		state_id_by_triplet: dict[tuple[int, int, int], int] = {}  # Not used in camera-based mode
		parsed_triplets: list[tuple[int, int, int]] = []  # Not used in camera-based mode
		
		# Dummy edges for saving (not meaningful in camera-based mode)
		pitch_vals = [0]
		angle_vals = [0]
		dist_vals = [0]
		pitch_edges = _edges_from_discrete_values(pitch_vals)
		az_edges = _edges_from_discrete_values(angle_vals)
		dist_edges = _edges_from_discrete_values(dist_vals)
		
		print(f"[消息] [HPCM] 相机模式: 状态数(每个相机一个状态): {len(state_bins)}；EnvMap 数: {len(hdr_bases_cpu)}")
	hpcm = HPCMTable(
		state_bins=state_bins,
		hdr_names=hdr_names,
		temperature=float(getattr(args, "hpcm_temperature", 1.0)),
		momentum=float(getattr(args, "hpcm_momentum", 0.9)),
		init_score=float(getattr(args, "hpcm_init_score", 0.0)),
		uniform_prob=float(getattr(args, "hpcm_uniform_prob", 0.0)),
	)
	# HPCM monitoring exports (summary/csv/plots) at save points
	hpcm_prev_scores: np.ndarray | None = None
	hpcm_stats_history: list[dict] = []

	rng = np.random.default_rng(seed=0)
	total_steps = int(getattr(args, "total_steps", 0))
	if total_steps <= 0:
		raise ValueError(f"--total_steps must be > 0, got {total_steps}")

	# --- Main step-based optimization loop ---
	pbar = tqdm(range(total_steps), total=total_steps, desc="HPCM (MIN-only)", ncols=120)
	# (NEW) Save detector visualization (with bboxes) on the NEXT step every N steps.
	det_vis_interval = int(getattr(args, "hpcm_det_vis_interval", 0))
	save_det_vis_next = False
	# Optional profiling (writes CSV under save_dir)
	profile_enabled = bool(getattr(args, "profile", False))
	profile_interval = int(getattr(args, "profile_interval", 50))
	profile_csv = save_dir / "profile_hpcm_step.csv"
	profile_header_written = False
	for step in pbar:
		do_profile = profile_enabled and profile_interval > 0 and (int(step) % profile_interval == 0)
		t_step0 = time.perf_counter() if do_profile else 0.0
		t_sampling = 0.0
		t_env = 0.0
		t_sh = 0.0
		t_forward = 0.0
		t_backward = 0.0
		t_opt = 0.0
		n_forward = 0
		n_loss_used = 0
		# HPCM batch: sample `batch_size` configs, compute per-sample loss, update table per-sample,
		# then backprop the MEAN loss across the batch to update albedo.
		resample_max = int(getattr(args, "hpcm_resample_max", 50))
		iter_for_render = render_global_step + step

		# If triggered, save bbox visualizations for THIS step (which is the "next batch").
		det_vis_dir_for_step = None
		if det_vis_interval > 0 and save_det_vis_next:
			det_vis_dir_for_step = save_dir / "det_vis" / f"step_{step:06d}"
			save_det_vis_next = False

		optimizer_min.zero_grad()
		losses_t: list[torch.Tensor] = []
		loss_vals: list[float] = []
		cls_vals: list[float] = []
		reg_vals: list[float] = []

		first_vis_data = None
		first_vis_name = None

		for bi in range(batch_size):
			# --- TEMPORARY GPU MONITOR ---
			if bi == 0 and step % 10 == 0:
				allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
				reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
				cache_len = len(relight_cache) if relight_cache is not None else 0
				print(f"\n[GPU-CHECK] Step: {step} | Allocated: {allocated:.1f}MB | Reserved: {reserved:.1f}MB | CacheItems: {cache_len}")
			# -----------------------------
			if do_profile:
				t0 = time.perf_counter()
			# Choose (camera, hdr) for this sample.
			# - hpcm: sample (state,hdr) by difficulty table, then pick a camera in that state.
			# - sequential: ablation: cycle cameras in a fixed order, and cycle hdr_id round-robin.
			sampling_mode = str(getattr(args, "hpcm_sampling", "hpcm")).lower()
			config_id = None
			state_id, hdr_id = None, None
			cam_idx = None
			if sampling_mode == "sequential":
				seq_k = (int(step) * int(batch_size) + int(bi)) % max(1, len(train_cameras))
				cam_idx = int(seq_k)
				if use_physical_states:
					triplet = parsed_triplets[cam_idx]
					state_id = int(state_id_by_triplet.get(tuple(triplet), -1))
					if state_id < 0:
						raise RuntimeError(f"Sequential sampling: cannot map triplet {triplet} to state_id.")
				else:
					# In camera-based mode, state_id = camera_idx
					state_id = int(cam_idx)
				hdr_id = int((int(step) * int(batch_size) + int(bi)) % max(1, len(hdr_bases_cpu)))
				config_id = hpcm.config_id(state_id, hdr_id)
			else:
				# sample a valid config_id (state must have at least 1 camera)
				for _ in range(max(1, resample_max)):
					cid = hpcm.sample_config(rng)
					sid, hid = hpcm.decode(cid)
					if cams_by_state[sid]:
						config_id = cid
						state_id, hdr_id = sid, hid
						break
				if config_id is None or state_id is None or hdr_id is None:
					available_state_ids = [i for i, lst in enumerate(cams_by_state) if lst]
					if not available_state_ids:
						raise RuntimeError("HPCM: no available states with cameras.")
					state_id = int(rng.choice(available_state_ids))
					hdr_id = int(rng.integers(0, len(hdr_bases_cpu)))
					config_id = hpcm.config_id(state_id, hdr_id)
			if do_profile:
				t_sampling += time.perf_counter() - t0

			# choose one camera for this state (HPCM mode); sequential already picked cam_idx above.
			if cam_idx is None:
				cam_indices = cams_by_state[state_id]
				cam_idx = int(rng.choice(cam_indices))
			cam = train_cameras[cam_idx]

			# switch envlight base for this sample
			with torch.no_grad():
				if do_profile:
					t0 = time.perf_counter()
				base_cpu = hdr_bases_cpu[hdr_id]
				gaussians.envlight.base = base_cpu.to(device)
				gaussians_original.envlight.base = base_cpu.to(device)
				try:
					gaussians.envlight.build_mips()
					gaussians_original.envlight.build_mips()
				except Exception:
					pass
				if do_profile:
					t_env += time.perf_counter() - t0
				# HDR->SH for LBM relighter conditioning
				if do_profile:
					t0 = time.perf_counter()
				if hdr_sh_cpu is not None:
					gaussians.hdr_sh_coeffs = hdr_sh_cpu[hdr_id].to(device)
				else:
					try:
						gaussians.hdr_sh_coeffs = base_cubemap_to_sh(gaussians.envlight.base, device)
					except Exception:
						gaussians.hdr_sh_coeffs = torch.zeros(27, dtype=torch.float32, device=device)
				if do_profile:
					t_sh += time.perf_counter() - t0

			# forward for this single camera
			if do_profile:
				t0 = time.perf_counter()
			total_loss, cls_loss, reg_loss, _, vis_data, _ = compute_batch_loss(
				[cam],
				gaussians,
				pipe,
				bg,
				iter_for_render,
				args,
				dataset,
				gaussians_original,
				relighter,
				detector,
				save_dir,
				0,
				step,
				det_vis_dir_for_step,
				hdr_id=hdr_id,
				relight_cache=relight_cache
			)
			if do_profile:
				t_forward += time.perf_counter() - t0
				n_forward += 1
			if total_loss is None:
				continue

			lv = float(total_loss.item())
			hpcm.update(config_id, lv)

			losses_t.append(total_loss)
			loss_vals.append(lv)
			cls_vals.append(float(cls_loss.item()) if cls_loss is not None else 0.0)
			reg_vals.append(float(reg_loss.item()) if reg_loss is not None else 0.0)
			n_loss_used += 1

			if first_vis_data is None and vis_data:
				first_vis_data = vis_data[0]
				first_vis_name = cam.image_name

		if not losses_t:
			continue

		if do_profile:
			t0 = time.perf_counter()
		mean_loss = torch.stack(losses_t, dim=0).mean()
		mean_loss.backward()
		if do_profile:
			t_backward += time.perf_counter() - t0

		mean_loss_val = float(mean_loss.item())
		mean_cls_val = float(np.mean(cls_vals)) if cls_vals else 0.0
		mean_reg_val = float(np.mean(reg_vals)) if reg_vals else 0.0

		logger.log_iteration(mean_loss_val, mean_cls_val, mean_reg_val)
		if do_profile:
			t0 = time.perf_counter()
		optimizer_min.step()
		with torch.no_grad():
			gaussians._albedo_init.data.clamp_(-0.5, 0.5)
		if do_profile:
			t_opt += time.perf_counter() - t0

		# Write profiling row
		if do_profile:
			try:
				t_total = time.perf_counter() - t_step0
				row = {
					"step": int(step),
					"batch_size": int(batch_size),
					"n_forward_calls": int(n_forward),
					"n_losses_used": int(n_loss_used),
					"total_s": float(t_total),
					"sampling_s": float(t_sampling),
					"env_switch_buildmips_s": float(t_env),
					"hdr_sh_s": float(t_sh),
					"forward_compute_batch_loss_s": float(t_forward),
					"backward_s": float(t_backward),
					"opt_step_s": float(t_opt),
				}
				# append
				write_header = (not profile_header_written) and (not profile_csv.exists())
				with profile_csv.open("a", encoding="utf-8", newline="") as f:
					w = csv.DictWriter(f, fieldnames=list(row.keys()))
					if write_header:
						w.writeheader()
						profile_header_written = True
					w.writerow(row)
				# print a compact summary
				print(
					"[PROFILE] step={step} total={total:.1f}ms | sample={sample:.1f} env+mips={env:.1f} sh={sh:.1f} "
					"forward={fwd:.1f} backward={bwd:.1f} opt={opt:.1f} | n_forward={nf} n_used={nu}".format(
						step=int(step),
						total=t_total * 1000.0,
						sample=t_sampling * 1000.0,
						env=t_env * 1000.0,
						sh=t_sh * 1000.0,
						fwd=t_forward * 1000.0,
						bwd=t_backward * 1000.0,
						opt=t_opt * 1000.0,
						nf=int(n_forward),
						nu=int(n_loss_used),
					)
				)
			except Exception as e:
				print(f"[警告] [PROFILE] 写 profile_hpcm_step.csv 失败: {e}")

		# Trigger saving bbox visualizations on the NEXT step every det_vis_interval steps.
		if det_vis_interval > 0 and ((step + 1) % det_vis_interval == 0):
			save_det_vis_next = True

		# Postfix: show mean loss and batch spread
		try:
			pbar.set_postfix_str(f"mean_loss={mean_loss_val:.4f} (min={min(loss_vals):.4f}, max={max(loss_vals):.4f})")
		except Exception:
			pbar.set_postfix_str(f"mean_loss={mean_loss_val:.4f}")

		# Optional visualization (first sample in this step)
		if bool(getattr(args, "save_visualizations", False)) and first_vis_data is not None:
			vis_save_dir = save_dir / "visualizations"
			vis_save_dir.mkdir(parents=True, exist_ok=True)
			cam_name = first_vis_name or "unknown"
			save_path = vis_save_dir / f"step_{step:06d}_{cam_name}.png"
			save_visualization_grid(save_path, first_vis_data)

		# 6) Periodic save of HPCM table
		save_int = int(getattr(args, "hpcm_save_interval", 0))
		if save_int > 0 and (step + 1) % save_int == 0:
			try:
				hpcm_path = save_dir / "hpcm_table.npz"
				hpcm.save_npz(hpcm_path, pitch_edges=pitch_edges, az_edges=az_edges, dist_edges=dist_edges)
				print(f"\n[消息] [HPCM] 已保存 difficulty table: {hpcm_path}")
				# Optional: keep step-suffixed snapshot for later comparison
				if bool(getattr(args, "hpcm_save_history_npz", False)):
					hpcm_hist_path = save_dir / f"hpcm_table_step_{step:06d}.npz"
					hpcm.save_npz(hpcm_hist_path, pitch_edges=pitch_edges, az_edges=az_edges, dist_edges=dist_edges)
					print(f"[消息] [HPCM] 已保存 difficulty snapshot: {hpcm_hist_path.name}")
				# Optional: export human-friendly summaries/plots
				if bool(getattr(args, "hpcm_export_summary", True)):
					hpcm_prev_scores = export_hpcm_monitor(
						save_dir=save_dir,
						step=int(step),
						hpcm=hpcm,
						pitch_vals=pitch_vals,
						angle_vals=angle_vals,
						dist_vals=dist_vals,
						args=args,
						prev_scores=hpcm_prev_scores,
						stats_history=hpcm_stats_history,
						pitch_edges=pitch_edges,
						az_edges=az_edges,
						dist_edges=dist_edges,
					)
			except Exception as e:
				print(f"\n[警告] [HPCM] 保存 difficulty table 失败: {e}")

	# Save final HPCM table
	try:
		hpcm_path = save_dir / "hpcm_table_final.npz"
		hpcm.save_npz(hpcm_path, pitch_edges=pitch_edges, az_edges=az_edges, dist_edges=dist_edges)
		print(f"[消息] [HPCM] 已保存最终 difficulty table: {hpcm_path}")
		# Also export a final readable snapshot
		if bool(getattr(args, "hpcm_export_summary", True)):
			hpcm_prev_scores = export_hpcm_monitor(
				save_dir=save_dir,
				step=int(total_steps),
				hpcm=hpcm,
				pitch_vals=pitch_vals,
				angle_vals=angle_vals,
				dist_vals=dist_vals,
				args=args,
				prev_scores=hpcm_prev_scores,
				stats_history=hpcm_stats_history,
				pitch_edges=pitch_edges,
				az_edges=az_edges,
				dist_edges=dist_edges,
			)
	except Exception as e:
		print(f"[警告] [HPCM] 保存最终 difficulty table 失败: {e}")


	# --- After training, plot curves ---
	logger.plot_iteration_losses()
	logger.plot_epoch_losses()
	logger.plot_asr_and_loss()
	logger.plot_ap_curve()

	# =================================================================================
	# 7. 最终评估与记录 (可选) - 修改为遍历所有检测器
	# =================================================================================
	if args.run_final_eval:
		print("\n[消息] 所有训练轮次完成。开始多检测器最终评估...")
		
		# Define datasets
		# Training already uses full set; avoid duplicates
		full_cameras = train_cameras_all

		# 在最终评估前恢复 envlight.base 为初始（checkpoint）版本（两套模型同步）
		selected_base_cpu = initial_env_base_cpu
		if selected_base_cpu is None:
			try:
				selected_base_cpu = gaussians.envlight.base.detach().cpu().clone()
			except Exception:
				selected_base_cpu = None
		if selected_base_cpu is not None:
			with torch.no_grad():
				try:
					gaussians.envlight.base = selected_base_cpu.to(device)
					gaussians.envlight.build_mips()
				except Exception:
					pass
				try:
					gaussians_original.envlight.base = selected_base_cpu.to(device)
					gaussians_original.envlight.build_mips()
				except Exception:
					pass

		# --- STAGE 1: Render Final Images for Offline Evaluation (using ori background only) ---
		final_full_img_dirs_mw = {}
		if bool(getattr(args, 'save_final_full_images_mw', True)):
			print("[消息] [最终评估] 正在渲染最终图片用于离线评估（使用 ori 目录的原始背景）...")
			final_full_img_dirs_mw = render_and_save_final_images_ori(
				full_cameras, gaussians, pipe, bg, args, dataset,
				gaussians_original, None, save_dir, 'full'
			)
		else:
			print("[消息] [最终评估] 已跳过渲染最终图片，将不会执行离线评估。")

		# --- (NEW) Also export evaluation_results_rpga.txt (same format as evaluate_img_rpga.py) ---
		# This evaluates final_*_images_* folders under save_dir and writes a unified summary txt.
		try:
			eval_txt_path = save_dir / 'evaluation_results_rpga.txt'
			script_path = Path('/workspace/RGA/evaluate_img_rpga.py')
			anno_dir = Path(dataset.source_path) / 'annos'
			mmdet_base = Path('/workspace/RGA/mmdet_files')

			if script_path.is_file() and anno_dir.is_dir() and mmdet_base.is_dir():
				cmd = [
					sys.executable, str(script_path),
					'--exp_dir', str(save_dir),
					'--anno_dir', str(anno_dir),
					'--mmdet_base', str(mmdet_base),
					'--device', str(args.device),
					'--all_detectors',
					'--score_thresh', str(getattr(args, 'score_thresh', 0.5)),
					'--output_file', str(eval_txt_path),
				]
				print(f"[消息] [最终评估] 正在生成汇总文件: {eval_txt_path.name}")
				result = subprocess.run(cmd, capture_output=True, text=True)
				if eval_txt_path.is_file():
					print(f"[消息] [最终评估] 已生成: {eval_txt_path}")
				else:
					print("[警告] [最终评估] evaluate_img_rpga.py 未生成 evaluation_results_rpga.txt（请检查脚本输出日志）。")
					if result.returncode != 0:
						print(f"    [调试信息] evaluate_img_rpga.py 执行失败，返回码: {result.returncode}")
						print(f"    --- STDOUT ---\n{result.stdout}")
						print(f"    --- STDERR ---\n{result.stderr}")
			else:
				print(
					f"[警告] [最终评估] 跳过生成 evaluation_results_rpga.txt："
					f"script={script_path.is_file()}, anno_dir={anno_dir.is_dir()}, mmdet_base={mmdet_base.is_dir()}"
				)
		except Exception as e:
			print(f"[警告] [最终评估] 生成 evaluation_results_rpga.txt 失败: {e}")

		# --- STAGE 2: Evaluate on saved images with all detectors ---
		if bool(getattr(args, "run_stage2_eval", False)):
			log_file_path = save_dir / 'training_log.txt'
			with open(log_file_path, 'a', encoding='utf-8') as f:
				f.write("\n\n==================================================\n")
				f.write(f"=== Final Cross-Detector Evaluation Results (Step {int(getattr(args, 'total_steps', 0))}) ===\n")
				f.write("==================================================\n")
				beijing_tz = timezone(timedelta(hours=8))
				f.write(f"Timestamp: {datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

			# Release the training detector
			del detector
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				
			base_path = Path('/workspace/RGA/mmdet_files')
			
			for det_name in DETECTOR_PATHS.keys():
				print(f"\n>>> [评估] 正在加载检测器: {det_name} ...")
				
				try:
					# Init detector
					cfg_path = str(base_path / DETECTOR_PATHS[det_name]['config'])
					ckpt_path = str(base_path / DETECTOR_PATHS[det_name]['ckpt'])
					
					curr_detector = init_detector(cfg_path, ckpt_path, device=device)
					if not hasattr(curr_detector, 'CLASSES'):
						curr_detector.CLASSES = coco_classes

					# Evaluate Directories from saved images (using ori background)
					mw_results = []
					if not final_full_img_dirs_mw:
						print("  - [结果] 未渲染任何图片，跳过评估。")
					else:
						for weather_name, img_dir in final_full_img_dirs_mw.items():
							asr_f_w, succ_f_w, total_f_w, ap50_f_w = evaluate_from_saved_images(
								curr_detector, img_dir, Path(dataset.source_path) / 'annos', args
							)
							mw_results.append((weather_name, 'full', asr_f_w, succ_f_w, total_f_w, ap50_f_w))
					
					# Log results
					print(f"  - [结果] 检测器: {det_name}")
					with open(log_file_path, 'a', encoding='utf-8') as f:
						f.write(f"Detector: {det_name}\n")
						# Write evaluation results
						if mw_results:
							f.write("  - [Evaluation on Full Set (ori background)]\n")
							for (wname, split, asr_w, succ_w, total_w, ap50_w) in mw_results:
								print(f"    * {wname}: ASR={asr_w:.4f}")
								f.write(f"    * {wname} [{split}] ASR: {asr_w:.4f} ({succ_w}/{total_w}), AP@0.5: {ap50_w:.4f}\n")
						else:
							f.write("  - No images were rendered for evaluation.\n")
						f.write("-" * 30 + "\n")
						
					# Cleanup
					del curr_detector
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						
				except Exception as e:
					print(f"[错误] 评估检测器 {det_name} 时发生错误: {e}")
					with open(log_file_path, 'a', encoding='utf-8') as f:
						f.write(f"Detector: {det_name} - FAILED: {e}\n")
						f.write("-" * 30 + "\n")
			
			print(f"\n[消息] 最终多检测器评估完成。结果已保存到: {log_file_path}")
		else:
			print("\n[消息] 已注释/禁用 Stage2（跨检测器离线评估）。如需开启请加参数: --run_stage2_eval")

	else:
		print("\n[消息] 根据设置，已跳过最终评估步骤。")


	# =================================================================================
	# 8. 保存最终模型
	# =================================================================================
	if gaussians is not None:
		gaussians.save_ply(str(save_dir / 'point_cloud_final.ply'))


if __name__ == '__main__':
	main()
