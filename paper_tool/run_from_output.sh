#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# - Given a subfolder name under RGA_output (e.g., 1229_222012_Beijing),
#   (default) build a 3-txt list like txt2csv.py's default:
#     paper_tool/evaluation_results_mw2.txt
#     <exp_dir>/evaluation_results_rpga*.txt  (prefer evaluation_results_rpga.txt)
#     paper_tool/evaluation_results_mw3.txt
#   and run txt2csv + csv2fig.
# - Optional: --all_txts to use ALL evaluation_results_*.txt under that folder (legacy behavior)
# - Convert to CSV via txt2csv.py
# - Plot figures via csv2fig.py
#
# Usage:
#   bash /workspace/RGA/paper_tool/run_from_output.sh 1229_222012_Beijing
#   bash /workspace/RGA/paper_tool/run_from_output.sh /workspace/RGA/RGA_output/1229_222012_Beijing
#
# Outputs:
#   /workspace/RGA/paper_tool/auto_out/<folder_name>/
#     - csv_results/*.csv
#     - fig_out/*.png (or other format if you tweak csv2fig args below)

ROOT_DIR="/workspace/RGA"
PAPER_DIR="${ROOT_DIR}/paper_tool"
DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/RGA_output"

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Missing argument."
  echo "Usage: $0 <output_subfolder_or_abs_path> [--all_txts] [--format png|pdf|svg] [--dpi N]"
  exit 2
fi

OUT_ARG="$1"
shift || true

# Optional args passthrough
FIG_FORMAT="png"
DPI="500"
USE_ALL_TXTS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all_txts)
      USE_ALL_TXTS="1"
      shift 1
      ;;
    --format)
      FIG_FORMAT="${2:-}"
      shift 2
      ;;
    --dpi)
      DPI="${2:-}"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

if [[ "${OUT_ARG}" == /* ]]; then
  EXP_DIR="${OUT_ARG}"
else
  EXP_DIR="${DEFAULT_OUTPUT_ROOT}/${OUT_ARG}"
fi

if [[ ! -d "${EXP_DIR}" ]]; then
  echo "[ERROR] Not a directory: ${EXP_DIR}"
  exit 1
fi

FOLDER_NAME="$(basename "${EXP_DIR}")"
AUTO_OUT_DIR="${PAPER_DIR}/auto_out/${FOLDER_NAME}"
CSV_OUT_DIR="${AUTO_OUT_DIR}/csv_results"
FIG_OUT_DIR="${AUTO_OUT_DIR}/fig_out"

mkdir -p "${CSV_OUT_DIR}" "${FIG_OUT_DIR}"

# Collect evaluation txts
TXT_FILES=()
if [[ "${USE_ALL_TXTS}" == "1" ]]; then
  # Legacy: take ALL evaluation_results_*.txt under exp dir
  mapfile -t TXT_FILES < <(find "${EXP_DIR}" -maxdepth 1 -type f -name "evaluation_results_*.txt" | sort)
else
  # Default: follow txt2csv.py's default "concatenation", but replace rpga2 with the TXT from THIS exp_dir.
  BASE1="${PAPER_DIR}/evaluation_results_mw2.txt"
  BASE3="${PAPER_DIR}/evaluation_results_mw3.txt"

  TARGET=""
  if [[ -f "${EXP_DIR}/evaluation_results_rpga.txt" ]]; then
    TARGET="${EXP_DIR}/evaluation_results_rpga.txt"
  else
    # fallback: pick the first evaluation_results_*.txt in exp_dir
    TARGET="$(find "${EXP_DIR}" -maxdepth 1 -type f -name "evaluation_results_*.txt" | sort | head -n 1 || true)"
  fi

  if [[ -f "${BASE1}" ]]; then TXT_FILES+=("${BASE1}"); fi
  if [[ -n "${TARGET}" && -f "${TARGET}" ]]; then TXT_FILES+=("${TARGET}"); fi
  if [[ -f "${BASE3}" ]]; then TXT_FILES+=("${BASE3}"); fi
fi

if [[ ${#TXT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No evaluation_results_*.txt found under: ${EXP_DIR}"
  echo "Expected something like: evaluation_results_rpga.txt"
  exit 1
fi

echo "[INFO] Using exp_dir: ${EXP_DIR}"
echo "[INFO] Found ${#TXT_FILES[@]} txt file(s):"
for f in "${TXT_FILES[@]}"; do
  echo "  - ${f}"
done

echo "[INFO] Running txt2csv.py -> ${CSV_OUT_DIR}"
python "${PAPER_DIR}/txt2csv.py" --input_txts "${TXT_FILES[@]}" --output_dir "${CSV_OUT_DIR}"

if [[ -f "${CSV_OUT_DIR}/all_tables_long.csv" ]]; then
  CSV_PATH="${CSV_OUT_DIR}/all_tables_long.csv"
  HAS_T5="$(
    python - "${CSV_PATH}" <<'PY'
import csv, sys
path = sys.argv[1]
has5 = False
with open(path, "r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get("table_id") == "5":
            has5 = True
            break
print("1" if has5 else "0")
PY
  )"
  if [[ "${HAS_T5}" == "1" ]]; then
    echo "[INFO] Detected Table 5 (angle) in CSV. Will also generate angle figures."
  else
    echo "[INFO] No Table 5 (angle) found in CSV. Will generate tables 1-4 figures only."
  fi
fi

echo "[INFO] Running csv2fig.py -> ${FIG_OUT_DIR}"
python "${PAPER_DIR}/csv2fig.py" --input_csv "${CSV_OUT_DIR}/all_tables_long.csv" --output_dir "${FIG_OUT_DIR}" --format "${FIG_FORMAT}" --dpi "${DPI}"

echo "[DONE] CSV: ${CSV_OUT_DIR}"
echo "[DONE] FIG: ${FIG_OUT_DIR}"


