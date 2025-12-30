#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hpcm_viewer.py

Offline viewer for HPCM difficulty tables saved by train_rel_attack_hpcm.py.

It reads `hpcm_table*.npz` and exports:
- summary txt (Top-K configs + global stats)
- stats plots (histogram)
- pitch-slice heatmaps of mean(score over HDR) and visit counts (if discrete grid is available)
- optional CSV exports

Example:
  python /workspace/RGA/paper_tool/hpcm_viewer.py \
    --input_npz /path/to/hpcm_table_final.npz \
    --out_dir /path/to/out \
    --topk 30
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("HPCM NPZ viewer")
    p.add_argument("--input_npz", type=str, required=True, help="Path to hpcm_table*.npz")
    p.add_argument("--out_dir", type=str, default="", help="Output directory (default: <npz_dir>/hpcm_view)")
    p.add_argument("--topk", type=int, default=20, help="Top-K configs to export")
    p.add_argument("--export_plots", default=True, action=argparse.BooleanOptionalAction, help="Export plots (hist/heatmaps).")
    p.add_argument("--export_full_csv", default=False, action=argparse.BooleanOptionalAction, help="Export full per-config CSV (can be large).")
    p.add_argument("--plot_max_pitches", type=int, default=12, help="Max number of pitch heatmaps to export (0 disables).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.input_npz)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (npz_path.parent / "hpcm_view")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path), allow_pickle=True)
    scores = np.asarray(data["scores"], dtype=np.float32)
    counts = np.asarray(data["counts"], dtype=np.int32) if "counts" in data.files else np.zeros_like(scores, dtype=np.int32)
    hdr_names = [str(x) for x in np.asarray(data["hdr_names"], dtype=object).tolist()] if "hdr_names" in data.files else []
    state_pitch = np.asarray(data["state_pitch"], dtype=np.int32)
    state_az = np.asarray(data["state_azimuth"], dtype=np.int32)
    state_dist = np.asarray(data["state_distance"], dtype=np.int32)
    temperature = float(np.asarray(data["temperature"]).reshape(-1)[0]) if "temperature" in data.files else 1.0

    # Infer shapes
    num_states = int(state_pitch.size)
    if num_states <= 0:
        raise RuntimeError("Invalid state arrays in npz.")
    if scores.size % num_states != 0:
        raise RuntimeError(f"scores size {scores.size} is not divisible by num_states {num_states}.")
    num_hdr = int(scores.size // num_states)
    scores_2d = scores.reshape(num_states, num_hdr)
    counts_2d = counts.reshape(num_states, num_hdr)

    # Global stats
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    q50 = _safe_percentile(scores, 50.0)
    q90 = _safe_percentile(scores, 90.0)
    q99 = _safe_percentile(scores, 99.0)
    visited = int(np.sum(counts > 0))
    visited_frac = float(visited / max(1, counts.size))

    try:
        probs = _softmax_np(scores.astype(np.float64) / max(1e-8, float(temperature)))
        ent = _entropy(probs)
    except Exception:
        ent = float("nan")

    topk = int(max(1, min(int(args.topk), scores.size)))
    top_idx = np.argsort(-scores.astype(np.float64))[:topk]

    def decode_config(config_id: int) -> tuple[int, int, int, int, str]:
        cid = int(config_id)
        sid = cid // num_hdr
        hid = cid % num_hdr
        pch = int(state_pitch[sid])
        az = int(state_az[sid])
        dist = int(state_dist[sid])
        hname = hdr_names[hid] if 0 <= hid < len(hdr_names) else str(hid)
        return pch, az, dist, hid, hname

    # Summary txt
    lines: list[str] = []
    lines.append(f"npz: {npz_path}")
    lines.append(f"num_states: {num_states}  num_hdr: {num_hdr}  num_configs: {scores.size}")
    lines.append(f"temperature: {temperature:.6g}")
    lines.append(f"score: mean={mean_score:.6f} std={std_score:.6f} min={min_score:.6f} p50={q50:.6f} p90={q90:.6f} p99={q99:.6f} max={max_score:.6f}")
    lines.append(f"visited_configs: {visited}/{counts.size} ({visited_frac*100.0:.2f}%)")
    lines.append(f"sampling_entropy: {ent:.6f}")
    lines.append("")
    lines.append(f"Top-{topk} configs by score (pitch, angle, distance, hdr, score, count):")
    for rank, cid in enumerate(top_idx, start=1):
        pch, az, dist, hid, hname = decode_config(int(cid))
        lines.append(
            f"{rank:02d}. pitch={pch:>4d} angle={az:>4d} distance={dist:>4d} hdr={hid:>3d}({hname}) "
            f"score={float(scores[int(cid)]):.6f} count={int(counts[int(cid)])}"
        )
    (out_dir / "hpcm_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # TopK CSV
    with (out_dir / "hpcm_topk.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "pitch", "angle", "distance", "hdr_id", "hdr_name", "score", "count"])
        for rank, cid in enumerate(top_idx, start=1):
            pch, az, dist, hid, hname = decode_config(int(cid))
            w.writerow([rank, pch, az, dist, hid, hname, float(scores[int(cid)]), int(counts[int(cid)])])

    # Full CSV (optional)
    if bool(args.export_full_csv):
        with (out_dir / "hpcm_full.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config_id", "state_id", "hdr_id", "pitch", "angle", "distance", "hdr_name", "score", "count"])
            for cid in range(scores.size):
                sid = cid // num_hdr
                hid = cid % num_hdr
                pch = int(state_pitch[sid])
                az = int(state_az[sid])
                dist = int(state_dist[sid])
                hname = hdr_names[hid] if 0 <= hid < len(hdr_names) else str(hid)
                w.writerow([cid, sid, hid, pch, az, dist, hname, float(scores[cid]), int(counts[cid])])

    # Plots
    if bool(args.export_plots):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Histogram
            fig = plt.figure(figsize=(8, 4.5))
            plt.hist(scores.astype(np.float64), bins=60)
            plt.title("HPCM score distribution")
            plt.xlabel("score")
            plt.ylabel("count")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "hpcm_score_hist.png", dpi=160)
            plt.close(fig)

            # Pitch-slice heatmaps of mean(score over HDR) if we can form a grid
            pitch_vals = sorted({int(x) for x in state_pitch.tolist()})
            angle_vals = sorted({int(x) for x in state_az.tolist()})
            dist_vals = sorted({int(x) for x in state_dist.tolist()})

            state_mean = scores_2d.mean(axis=1).astype(np.float64)
            state_cnt = counts_2d.sum(axis=1).astype(np.int64)

            P, A, D = len(pitch_vals), len(angle_vals), len(dist_vals)

            # Build mapping (works even if not full cartesian product)
            sid_by_key: dict[tuple[int, int, int], int] = {}
            for sid in range(num_states):
                key = (int(state_pitch[sid]), int(state_az[sid]), int(state_dist[sid]))
                sid_by_key[key] = sid

            max_pitches = int(args.plot_max_pitches)
            if max_pitches != 0 and P > 0 and A > 0 and D > 0:
                if max_pitches > 0 and P > max_pitches:
                    idxs = np.linspace(0, P - 1, max_pitches).round().astype(int).tolist()
                    pitch_idxs = sorted(set(int(i) for i in idxs))
                else:
                    pitch_idxs = list(range(P))

                hm_dir = out_dir / "heatmaps"
                hm_dir.mkdir(parents=True, exist_ok=True)

                for ip in pitch_idxs:
                    pch = int(pitch_vals[ip])
                    hm = np.full((A, D), np.nan, dtype=np.float64)
                    hc = np.full((A, D), np.nan, dtype=np.float64)
                    for ia, az in enumerate(angle_vals):
                        for idd, dist in enumerate(dist_vals):
                            sid = sid_by_key.get((pch, int(az), int(dist)))
                            if sid is None:
                                continue
                            hm[ia, idd] = float(state_mean[sid])
                            hc[ia, idd] = float(state_cnt[sid])

                    # mean heatmap
                    fig, ax = plt.subplots(figsize=(7.2, 4.8))
                    im = ax.imshow(np.ma.masked_invalid(hm), aspect="auto")
                    ax.set_title(f"mean(score over HDR) | pitch={pch}")
                    ax.set_xlabel("distance")
                    ax.set_ylabel("angle")
                    ax.set_xticks(list(range(D)))
                    ax.set_xticklabels([str(int(v)) for v in dist_vals], rotation=45, ha="right", fontsize=8)
                    ax.set_yticks(list(range(A)))
                    ax.set_yticklabels([str(int(v)) for v in angle_vals], fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(hm_dir / f"hpcm_mean_heatmap_pitch{pch}.png", dpi=160)
                    plt.close(fig)

                    # count heatmap
                    fig, ax = plt.subplots(figsize=(7.2, 4.8))
                    im = ax.imshow(np.ma.masked_invalid(hc), aspect="auto")
                    ax.set_title(f"visit_count(sum over HDR) | pitch={pch}")
                    ax.set_xlabel("distance")
                    ax.set_ylabel("angle")
                    ax.set_xticks(list(range(D)))
                    ax.set_xticklabels([str(int(v)) for v in dist_vals], rotation=45, ha="right", fontsize=8)
                    ax.set_yticks(list(range(A)))
                    ax.set_yticklabels([str(int(v)) for v in angle_vals], fontsize=8)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(hm_dir / f"hpcm_count_heatmap_pitch{pch}.png", dpi=160)
                    plt.close(fig)
        except Exception as e:
            print(f"[WARN] failed to export plots: {e}")

    print(f"[OK] exported to: {out_dir}")


if __name__ == "__main__":
    main()


