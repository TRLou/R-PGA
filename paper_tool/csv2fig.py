#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv2fig.py

Convert the "long/tidy" CSV produced by txt2csv (recommended: all_tables_long.csv)
into publication-ready figures:

Table 1 (pitch): line plot, one curve per method
Table 2 (distance): line plot, one curve per method
Table 3 (weather): grouped bar chart, one group per weather, methods compared within each group
Table 4 (detector): grouped bar chart, one group per detector, methods compared within each group
Table 5 (angle): line plot, one curve per method

ASR and AP@0.5 are plotted separately: 2 figures per table.
Note: Table 5 (angle) is OPTIONAL. If table_id==5 is absent in CSV, it will be skipped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

# =================================================================================
# Plotting Constants
# =================================================================================

# User-defined color palette (RGB values from 0-255, converted to 0-1)
# Reversed direction: Blue -> ... -> Red
COLOR_PALETTE = [
    (93/255.0, 173/255.0, 226/255.0),   # Blue (was idx 5)
    (72/255.0, 79/255.0, 152/255.0),    # Dark Blue (was idx 9)
    (145/255.0, 223/255.0, 208/255.0),  # Cyan (was idx 4)
    (82/255.0, 190/255.0, 128/255.0),   # Green (was idx 3)
    (255/255.0, 255/255.0, 133/255.0),  # Pale Yellow (was idx 10)
    (246/255.0, 218/255.0, 101/255.0),  # Yellow (was idx 2)
    (245/255.0, 176/255.0, 65/255.0),   # Orange (was idx 1)
    (255/255.0, 188/255.0, 167/255.0),  # Pink (was idx 8)
    (163/255.0, 105/255.0, 189/255.0),  # Purple (was idx 6)
    (213/255.0, 105/255.0, 93/255.0),   # Red (was idx 0)
]

# User-defined method order
# Note: DAS is excluded by default (filtered in txt2csv.py)
METHOD_ORDER = [
    'ORI', 'FCA', 'DTA', 'ACTIVE', 'RAUCA',
    'GCAC', 'GRAC', 'RAUCA-E2E', 'PGA', 'R-PGA'
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MW2 summary CSV into figures (8 figs: 4 tables x 2 metrics)")
    p.add_argument(
        "--input_csv",
        default="./csv_results/all_tables_long.csv",
        help="Input long/tidy CSV",
    )
    p.add_argument(
        "--output_dir",
        default="./fig_out",
        help="Output figure directory (default: ./fig_out relative to paper_tool/)",
    )
    p.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="DPI for PNG output",
    )
    p.add_argument(
        "--metric",
        default="both",
        choices=["asr", "ap50", "both"],
        help="Which metric(s) to plot: asr / ap50 / both",
    )
    p.add_argument(
        "--style",
        default="default",
        help="matplotlib style (e.g., 'seaborn-v0_8-whitegrid')",
    )
    p.add_argument(
        "--font",
        default="",
        help="Optional: set a specific font family name (e.g., 'DejaVu Sans'). Leave empty for default.",
    )
    return p.parse_args()


def _try_set_chinese_font(font_name: str) -> None:
    if not font_name:
        return
    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["axes.unicode_minus"] = False


def read_long_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize types
            r["table_id"] = str(r.get("table_id", "")).strip()
            method_str = str(r.get("method", "")).strip()
            # Normalize method names for consistency
            if method_str.lower() == 'raucae2e':
                method_str = 'RAUCA-E2E'
            else:
                # Convert to uppercase for display
                method_str = method_str.upper()
            r["method"] = method_str
            r["variable_type"] = str(r.get("variable_type", "")).strip()
            r["variable_value"] = str(r.get("variable_value", "")).strip()
            try:
                r["mean_asr"] = float(r.get("mean_asr", "nan"))
            except (ValueError, TypeError):
                r["mean_asr"] = float("nan")
            try:
                r["mean_ap50"] = float(r.get("mean_ap50", "nan"))
            except (ValueError, TypeError):
                r["mean_ap50"] = float("nan")
            try:
                r["total"] = int(float(r.get("total", "0")))
            except (ValueError, TypeError):
                r["total"] = 0
            rows.append(r)
    return rows


def _get_ordered_methods(rows: list[dict]) -> list[str]:
    present_methods = {r["method"] for r in rows if r.get("method")}
    # Keep predefined order, but only for methods present in data
    ordered_methods = [m for m in METHOD_ORDER if m in present_methods]
    # Add any other methods from data that were not in our predefined list, sorted at the end
    other_methods = sorted(list(present_methods - set(ordered_methods)))
    return ordered_methods + other_methods


def _parse_pitch_value(v: str) -> int | None:
    # pitch5 / Pitch5 ...
    v = v.strip()
    if v.lower().startswith("pitch"):
        try:
            return int(v[5:])
        except Exception:
            return None
    return None


def _parse_distance_value(v: str) -> int | None:
    v = v.strip()
    if v.lower().startswith("distance"):
        try:
            return int(v[8:])
        except Exception:
            return None
    return None


def _parse_angle_value(v: str) -> int | None:
    # angle240 / Angle-30 ...
    v = v.strip()
    if v.lower().startswith("angle"):
        try:
            return int(v[5:])
        except Exception:
            return None
    return None


def _group_by_method(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, out_path: Path, fmt: str, dpi: int) -> None:
    if fmt == "png":
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_line_table(
    rows: List[dict],
    *,
    x_kind: str,  # "pitch" / "distance" / "angle"
    metric_key: str,  # "mean_asr" / "mean_ap50"
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    methods = _get_ordered_methods(rows)
    by_method = _group_by_method(rows)
    # Create color map, ensuring R-PGA gets red color
    red_color = (213/255.0, 105/255.0, 93/255.0)  # Red color
    color_map = {method: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, method in enumerate(methods)}
    # Ensure R-PGA uses red color
    if 'R-PGA' in methods:
        color_map['R-PGA'] = red_color

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for m in methods:
        pts = by_method.get(m, [])
        xs: List[int] = []
        ys: List[float] = []
        for r in pts:
            if x_kind == "pitch":
                xv = _parse_pitch_value(r["variable_value"])
            elif x_kind == "distance":
                xv = _parse_distance_value(r["variable_value"])
            else:
                xv = _parse_angle_value(r["variable_value"])
            if xv is None:
                continue
            xs.append(xv)
            ys.append(float(r[metric_key]))

        # Sort points by x-value
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        if not pairs:
            continue
        xs2, ys2 = zip(*pairs)
        ax.plot(xs2, ys2, marker="o", linewidth=2.0, markersize=4.5, label=m, color=color_map.get(m))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncols=2, fontsize=9, frameon=True)

    _save_fig(fig, out_path, fmt, dpi)


def plot_grouped_bar_table(
    rows: List[dict],
    *,
    metric_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    methods = _get_ordered_methods(rows)
    categories = sorted({r["variable_value"] for r in rows})
    # Create color map, ensuring R-PGA gets red color
    red_color = (213/255.0, 105/255.0, 93/255.0)  # Red color
    color_map = {method: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, method in enumerate(methods)}
    # Ensure R-PGA uses red color
    if 'R-PGA' in methods:
        color_map['R-PGA'] = red_color

    # lookup[(method, category)] = metric
    lookup: Dict[Tuple[str, str], float] = {}
    for r in rows:
        lookup[(r["method"], r["variable_value"])] = float(r[metric_key])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    n_cat = len(categories)
    n_m = max(1, len(methods))
    total_width = 0.82
    bar_w = total_width / n_m

    x_base = list(range(n_cat))
    for j, m in enumerate(methods):
        offsets = [x + (j - (n_m - 1) / 2) * bar_w for x in x_base]
        ys = [lookup.get((m, c), float("nan")) for c in categories]
        ax.bar(offsets, ys, width=bar_w * 0.98, label=m, color=color_map.get(m))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncols=2, fontsize=9, frameon=True)

    _save_fig(fig, out_path, fmt, dpi)


def main() -> None:
    args = parse_args()
    try:
        plt.style.use(args.style)
    except Exception:
        # Fallback to default if style not found
        pass
    _try_set_chinese_font(args.font)

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(output_dir)

    rows = read_long_csv(input_csv)

    # Split data by table_id
    t1 = [r for r in rows if r.get("table_id") == "1"]
    t2 = [r for r in rows if r.get("table_id") == "2"]
    t3 = [r for r in rows if r.get("table_id") == "3"]
    t4 = [r for r in rows if r.get("table_id") == "4"]
    t5 = [r for r in rows if r.get("table_id") == "5"]
    if not t5:
        print("[INFO] Table 5 (angle) not found in CSV (table_id==5). Will plot tables 1-4 only.")

    metrics: List[Tuple[str, str, str]] = []
    # (metric_key, metric_name, ylabel)
    if args.metric in ("asr", "both"):
        metrics.append(("mean_asr", "ASR", "Mean ASR"))
    if args.metric in ("ap50", "both"):
        metrics.append(("mean_ap50", "AP@0.5", "Mean AP@0.5"))

    for metric_key, metric_name, ylabel in metrics:
        if t1:
            plot_line_table(
                t1,
                x_kind="pitch",
                metric_key=metric_key,
                title=f"Pitch comparison - {metric_name}",
                xlabel="pitch",
                ylabel=ylabel,
                out_path=output_dir / f"table1_pitch_{metric_name}.{args.format}",
                fmt=args.format,
                dpi=args.dpi,
            )
        if t2:
            plot_line_table(
                t2,
                x_kind="distance",
                metric_key=metric_key,
                title=f"Distance comparison - {metric_name}",
                xlabel="distance",
                ylabel=ylabel,
                out_path=output_dir / f"table2_distance_{metric_name}.{args.format}",
                fmt=args.format,
                dpi=args.dpi,
            )
        if t3:
            plot_grouped_bar_table(
                t3,
                metric_key=metric_key,
                title=f"Weather comparison - {metric_name}",
                xlabel="weather",
                ylabel=ylabel,
                out_path=output_dir / f"table3_weather_{metric_name}.{args.format}",
                fmt=args.format,
                dpi=args.dpi,
            )
        if t4:
            plot_grouped_bar_table(
                t4,
                metric_key=metric_key,
                title=f"Detector comparison - {metric_name}",
                xlabel="detector",
                ylabel=ylabel,
                out_path=output_dir / f"table4_detector_{metric_name}.{args.format}",
                fmt=args.format,
                dpi=args.dpi,
            )
        if t5:
            plot_line_table(
                t5,
                x_kind="angle",
                metric_key=metric_key,
                title=f"Angle comparison - {metric_name}",
                xlabel="angle",
                ylabel=ylabel,
                out_path=output_dir / f"table5_angle_{metric_name}.{args.format}",
                fmt=args.format,
                dpi=args.dpi,
            )

    print(f"[INFO] Done: read {len(rows)} rows from {input_csv} and saved figures to {output_dir}")


if __name__ == "__main__":
    main()
