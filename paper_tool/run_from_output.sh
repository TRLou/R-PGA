#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# - Given a subfolder name under RGA_output (e.g., 1229_222012_Beijing),
#   (default) build a txt list:
#     paper_tool/evaluation_results_mw2_5table.txt
#     RGA_output/PGA/evaluation_results_*.txt  (PGA method, second to last)
#     RGA_output/paper_exp/evaluation_results_rpga*.txt  (R-PGA method, last)
#   Note: DAS results are automatically filtered out.
#   and run txt2csv + csv2fig.
# - Optional: --all_txts to use ALL evaluation_results_*.txt under that folder (legacy behavior)
# - Convert to CSV via txt2csv.py
# - Plot figures via csv2fig.py
#
# Usage:
#   bash /workspace/RGA/paper_tool/run_from_output.sh [output_name] [--all_txts] [--format png|pdf|svg] [--dpi N]
#   
#   Default mode (no --all_txts):
#     - Uses fixed paths: mw2_5table.txt, PGA/, paper_exp/
#     - output_name is optional (default: "default")
#     - Example: bash paper_tool/run_from_output.sh
#     - Example: bash paper_tool/run_from_output.sh my_output
#
#   Legacy mode (--all_txts):
#     - Uses all txt files under the specified folder
#     - output_name is required
#     - Example: bash paper_tool/run_from_output.sh 1229_222012_Beijing --all_txts
#
# Outputs:
#   /workspace/RGA/paper_tool/auto_out/<output_name>/
#     - csv_results/*.csv
#     - fig_out/*.png (or other format if you tweak csv2fig args below)

ROOT_DIR="/workspace/RGA"
PAPER_DIR="${ROOT_DIR}/paper_tool"
DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/RGA_output"

# Parse arguments - output_name is now optional
OUT_ARG=""
USE_ALL_TXTS="0"
FIG_FORMAT="png"
DPI="500"

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
    --*)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
    *)
      # First non-option argument is the output name
      if [[ -z "${OUT_ARG}" ]]; then
        OUT_ARG="$1"
      else
        echo "[ERROR] Unexpected argument: $1"
        exit 2
      fi
      shift 1
      ;;
  esac
done

# Set default output name if not provided
if [[ -z "${OUT_ARG}" ]]; then
  if [[ "${USE_ALL_TXTS}" == "1" ]]; then
    echo "[ERROR] --all_txts mode requires an output folder name."
    echo "Usage: $0 <output_subfolder_or_abs_path> --all_txts [--format png|pdf|svg] [--dpi N]"
    exit 2
  else
    OUT_ARG="default"
  fi
fi

# Determine output folder name and exp_dir
# In default mode, OUT_ARG is just used for output directory name
# In --all_txts mode, OUT_ARG is used to find input files
if [[ "${USE_ALL_TXTS}" == "1" ]]; then
  # Legacy mode: OUT_ARG must be a valid directory
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
else
  # Default mode: OUT_ARG is just the output name, no directory check needed
  FOLDER_NAME="${OUT_ARG}"
  EXP_DIR=""  # Not used in default mode
fi
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
  # Default: use fixed paths for PGA and R-PGA, ignoring DAS results
  BASE1="${PAPER_DIR}/evaluation_results_mw2_5table.txt"
  
  # PGA folder: find evaluation_results_*.txt
  PGA_DIR="${DEFAULT_OUTPUT_ROOT}/PGA"
  PGA_TARGET=""
  if [[ -d "${PGA_DIR}" ]]; then
    PGA_TARGET="$(find "${PGA_DIR}" -maxdepth 1 -type f -name "evaluation_results_*.txt" | sort | head -n 1 || true)"
  fi
  
  # paper_exp folder: find evaluation_results_rpga*.txt (prefer evaluation_results_rpga.txt)
  PAPER_EXP_DIR="${DEFAULT_OUTPUT_ROOT}/paper_exp"
  RPGA_TARGET=""
  if [[ -d "${PAPER_EXP_DIR}" ]]; then
    if [[ -f "${PAPER_EXP_DIR}/evaluation_results_rpga.txt" ]]; then
      RPGA_TARGET="${PAPER_EXP_DIR}/evaluation_results_rpga.txt"
    else
      RPGA_TARGET="$(find "${PAPER_EXP_DIR}" -maxdepth 1 -type f -name "evaluation_results_rpga*.txt" | sort | head -n 1 || true)"
    fi
  fi

  if [[ -f "${BASE1}" ]]; then TXT_FILES+=("${BASE1}"); fi
  if [[ -n "${PGA_TARGET}" && -f "${PGA_TARGET}" ]]; then TXT_FILES+=("${PGA_TARGET}"); fi
  if [[ -n "${RPGA_TARGET}" && -f "${RPGA_TARGET}" ]]; then TXT_FILES+=("${RPGA_TARGET}"); fi
fi

if [[ ${#TXT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No evaluation_results_*.txt files found."
  if [[ "${USE_ALL_TXTS}" == "1" ]]; then
    echo "Expected files under: ${EXP_DIR}"
  else
    echo "Expected files:"
    echo "  - ${BASE1}"
    echo "  - ${PGA_DIR}/evaluation_results_*.txt (PGA)"
    echo "  - ${PAPER_EXP_DIR}/evaluation_results_rpga*.txt (R-PGA)"
  fi
  exit 1
fi

if [[ "${USE_ALL_TXTS}" == "1" ]]; then
  echo "[INFO] Using exp_dir: ${EXP_DIR}"
else
  echo "[INFO] Using default fixed paths (PGA and paper_exp folders)"
fi
echo "[INFO] Found ${#TXT_FILES[@]} txt file(s):"
for f in "${TXT_FILES[@]}"; do
  echo "  - ${f}"
done

echo "[INFO] Running txt2csv.py -> ${CSV_OUT_DIR} (DAS results will be filtered out)"
python "${PAPER_DIR}/txt2csv.py" --input_txts "${TXT_FILES[@]}" --output_dir "${CSV_OUT_DIR}" --exclude_methods DAS

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


