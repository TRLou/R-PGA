#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 evaluate_img_mw2.py 输出的中文 txt 汇总结果转换为可直接导入的 CSV。

输出：
1) all_tables_long.csv：长表（推荐），每行一条 (method, variable_type, variable_value) 的统计
2) 每张表各自的长表：table1_pitch_long.csv / table2_distance_long.csv / table3_weather_long.csv / table4_detector_long.csv
   （可选）若 txt 中存在 表5：控制变量【方位角 angle】，也会输出 table5_angle_long.csv
3) 每张表各自的宽表（pivot）：*_wide_asr.csv 与 *_wide_ap50.csv（行=方法，列=变量取值）
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TABLE_HEADER_RE = re.compile(r"^=+ 表\s*(\d+)\s*：控制变量【(.+?)】=+$")
METHOD_RE = re.compile(r"^【方法：(.+?)】$")

PITCH_RE = re.compile(r"^变量俯仰角：pitch(\d+)，(.+?)方法的平均ASR是([0-9.]+)，平均AP@0\.5是([0-9.]+)（成功(\d+)/(\d+)）$")
DIST_RE = re.compile(r"^变量距离：distance(\d+)，(.+?)方法的平均ASR是([0-9.]+)，平均AP@0\.5是([0-9.]+)（成功(\d+)/(\d+)）$")
WEATHER_RE = re.compile(r"^变量天气：(.+?)，(.+?)方法的平均ASR是([0-9.]+)，平均AP@0\.5是([0-9.]+)（成功(\d+)/(\d+)）$")
DETECTOR_RE = re.compile(r"^变量检测器：(.+?)，(.+?)方法的平均ASR是([0-9.]+)，平均AP@0\.5是([0-9.]+)（成功(\d+)/(\d+)）$")
ANGLE_RE = re.compile(r"^变量方位角：angle(-?\d+)，(.+?)方法的平均ASR是([0-9.]+)，平均AP@0\.5是([0-9.]+)（成功(\d+)/(\d+)）$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将评估中文 txt 汇总结果转换为 CSV（支持单文件与多文件合并）")
    p.add_argument(
        "--input_txt",
        default="evaluation_results_mw2.txt",
        help="输入 txt（单文件）。若指定了 --input_txts/--input_glob，则该参数可忽略。",
    )
    p.add_argument(
        "--input_txts",
        nargs="*",
        default=["evaluation_results_mw2.txt", "evaluation_results_rpga2.txt", "evaluation_results_mw3.txt"],
        help="输入 txt 列表（多个文件合并）。例如：--input_txts a.txt b.txt c.txt",
    )
    p.add_argument(
        "--input_glob",
        default="",
        help="用 glob 批量匹配 txt（会合并）。例如：--input_glob './RGA_output/*/evaluation_results_*.txt'",
    )
    p.add_argument(
        "--output_dir",
        default="./csv_results",
        help="输出目录（会写多个 csv 文件）",
    )
    p.add_argument(
        "--run_name",
        default="",
        help="可选：覆盖单文件模式下的 run 名称（默认用 input_txt 的 stem）。多文件模式下会忽略。",
    )
    return p.parse_args()


def _infer_variable_type(table_title: str) -> str:
    # table_title 示例：俯仰角 pitch / 距离 distance / 天气 weather / 检测器 detector
    if "俯仰角" in table_title or "pitch" in table_title:
        return "pitch"
    if "距离" in table_title or "distance" in table_title:
        return "distance"
    if "方位角" in table_title or "angle" in table_title or "azimuth" in table_title:
        return "angle"
    if "天气" in table_title or "weather" in table_title:
        return "weather"
    if "检测器" in table_title or "detector" in table_title:
        return "detector"
    return table_title.strip()


def parse_txt(txt_path: Path, *, run: str) -> list[dict]:
    rows: list[dict] = []
    cur_table_id: str | None = None
    cur_table_title: str | None = None
    cur_var_type: str | None = None
    cur_method: str | None = None

    for raw in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        m = TABLE_HEADER_RE.match(line)
        if m:
            cur_table_id = m.group(1)
            cur_table_title = m.group(2)
            cur_var_type = _infer_variable_type(cur_table_title)
            cur_method = None
            continue

        m = METHOD_RE.match(line)
        if m:
            cur_method = m.group(1)
            continue

        if cur_table_id is None or cur_var_type is None or cur_method is None:
            continue

        # 按表类型选择对应解析器
        if cur_var_type == "pitch":
            mm = PITCH_RE.match(line)
            if not mm:
                continue
            variable_value = f"pitch{mm.group(1)}"
            asr = float(mm.group(3))
            ap50 = float(mm.group(4))
            succ = int(mm.group(5))
            total = int(mm.group(6))
        elif cur_var_type == "distance":
            mm = DIST_RE.match(line)
            if not mm:
                continue
            variable_value = f"distance{mm.group(1)}"
            asr = float(mm.group(3))
            ap50 = float(mm.group(4))
            succ = int(mm.group(5))
            total = int(mm.group(6))
        elif cur_var_type == "weather":
            mm = WEATHER_RE.match(line)
            if not mm:
                continue
            variable_value = mm.group(1)
            asr = float(mm.group(3))
            ap50 = float(mm.group(4))
            succ = int(mm.group(5))
            total = int(mm.group(6))
        elif cur_var_type == "detector":
            mm = DETECTOR_RE.match(line)
            if not mm:
                continue
            variable_value = mm.group(1)
            asr = float(mm.group(3))
            ap50 = float(mm.group(4))
            succ = int(mm.group(5))
            total = int(mm.group(6))
        elif cur_var_type == "angle":
            mm = ANGLE_RE.match(line)
            if not mm:
                continue
            variable_value = f"angle{mm.group(1)}"
            asr = float(mm.group(3))
            ap50 = float(mm.group(4))
            succ = int(mm.group(5))
            total = int(mm.group(6))
        else:
            # 未知表类型，跳过
            continue

        rows.append(
            {
                "run": run,
                "source_txt": str(txt_path),
                "table_id": cur_table_id,
                "table_title": cur_table_title,
                "variable_type": cur_var_type,
                "variable_value": variable_value,
                "method": cur_method,
                "mean_asr": asr,
                "mean_ap50": ap50,
                "successful": succ,
                "total": total,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def pivot_wide(
    rows: list[dict],
    *,
    index_key: str,
    column_key: str,
    value_key: str,
) -> tuple[list[str], list[dict]]:
    """
    生成宽表：
      - header: [index_key] + sorted(unique(column_key))
      - out_rows: 每行是 {index_key: ..., col1: value, col2: value, ...}
    """
    idx_vals = sorted({r[index_key] for r in rows})
    col_vals = sorted({r[column_key] for r in rows})

    lookup: dict[tuple[str, str], object] = {}
    for r in rows:
        lookup[(str(r[index_key]), str(r[column_key]))] = r[value_key]

    header = [index_key] + col_vals
    out_rows: list[dict] = []
    for iv in idx_vals:
        row = {index_key: iv}
        for cv in col_vals:
            row[cv] = lookup.get((str(iv), str(cv)), "")
        out_rows.append(row)
    return header, out_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    # Resolve input txts (single or multi)
    input_paths: list[Path] = []
    if args.input_txts is not None and len(args.input_txts) > 0:
        input_paths.extend([Path(p) for p in args.input_txts])
    if str(args.input_glob).strip():
        input_paths.extend(sorted(Path().glob(str(args.input_glob))))
    if not input_paths:
        input_paths = [Path(args.input_txt)]

    # De-duplicate while preserving order
    seen = set()
    deduped: list[Path] = []
    for p in input_paths:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        deduped.append(p)
    input_paths = deduped

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"找不到输入 txt：{[str(p) for p in missing]}")

    rows: list[dict] = []
    multi_mode = len(input_paths) > 1
    for p in input_paths:
        if (not multi_mode) and str(args.run_name).strip():
            run = str(args.run_name).strip()
        else:
            run = p.stem
        rows.extend(parse_txt(p, run=run))

    if not rows:
        raise RuntimeError("未解析到任何数据行：请确认输入 txt 格式与脚本匹配。")

    fieldnames = [
        "run",
        "source_txt",
        "table_id",
        "table_title",
        "variable_type",
        "variable_value",
        "method",
        "mean_asr",
        "mean_ap50",
        "successful",
        "total",
    ]
    write_csv(out_dir / "all_tables_long.csv", rows, fieldnames)

    # 分表输出（长表 + 宽表）
    by_table: dict[str, list[dict]] = {}
    for r in rows:
        by_table.setdefault(str(r["table_id"]), []).append(r)

    table_name_map = {
        "1": "pitch",
        "2": "distance",
        "3": "weather",
        "4": "detector",
        "5": "angle",
    }

    for tid, sub in sorted(by_table.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        suffix = table_name_map.get(tid, f"table{tid}")
        write_csv(out_dir / f"table{tid}_{suffix}_long.csv", sub, fieldnames)

        # 宽表：ASR / AP50
        # 多文件合并时，为避免不同 run 下 method 同名被覆盖，使用 (run, method) 作为行索引。
        idx_key = "method" if not multi_mode else "run_method"
        sub_for_pivot = sub
        if multi_mode:
            sub_for_pivot = []
            for r in sub:
                rr = dict(r)
                rr["run_method"] = f"{rr.get('run')}::{rr.get('method')}"
                sub_for_pivot.append(rr)

        header_asr, wide_asr = pivot_wide(sub_for_pivot, index_key=idx_key, column_key="variable_value", value_key="mean_asr")
        header_ap, wide_ap = pivot_wide(sub_for_pivot, index_key=idx_key, column_key="variable_value", value_key="mean_ap50")

        # 这里宽表行是 dict[str, object]，用 csv.DictWriter 输出
        write_csv(out_dir / f"table{tid}_{suffix}_wide_asr.csv", wide_asr, header_asr)
        write_csv(out_dir / f"table{tid}_{suffix}_wide_ap50.csv", wide_ap, header_ap)

    print(f"[INFO] 转换完成：共解析 {len(rows)} 行，已输出到目录：{out_dir}")


if __name__ == "__main__":
    main()


