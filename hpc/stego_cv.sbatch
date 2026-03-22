#!/bin/bash
#SBATCH --job-name=stego_cv
#SBATCH --output=stego_cv-%j.log
#SBATCH --error=stego_cv-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
##SBATCH --account=<YOUR_PROJECT_ID>

set -euo pipefail

# ---- TMPDIR policy (НЕ /tmp) ----
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-}}"
if [[ -z "${TMPDIR}" ]]; then
  echo "TMPDIR is not set (and SLURM_TMPDIR is empty). Ask admins what to use." >&2
  exit 1
fi
export JOB_TMPDIR="${TMPDIR%/}/stego_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMPDIR"
trap 'rm -rf "$JOB_TMPDIR"' EXIT
export TMPDIR="$JOB_TMPDIR"

module purge
module load Python/Anaconda 2>/dev/null || module load Python 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR"

# ---- Python env ----
export PYTHONPATH="$SLURM_SUBMIT_DIR/flipad/src:$PYTHONPATH"

# ---- S3 creds (нужны для upload результатов) ----
# ВСТАВЬ СЮДА КЛЮЧИ (на сервере это ок, но НЕ коммить)
export AWS_ACCESS_KEY_ID="ВАШ_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="ВАШ_SECRET_ACCESS_KEY"
export AWS_REGION="ru-central1"
export AWS_DEFAULT_REGION="ru-central1"
export ENDPOINT="https://storage.yandexcloud.net"
export BUCKET="stegopractice"

# Укажи cache run dir (после выполнения stego_cache job):
RUN_DIR="${RUN_DIR:-runs/cache_<CACHE_JOB_ID>}"

python3 -u hpc/run_stego_vs_sd_dct.py cv \
  --cache-dir "$RUN_DIR" \
  --out-dir "$RUN_DIR/cv_k5" \
  --k 5 \
  --seed 20260320 \
  --val-size 10000 \
  --train-per-class 130000 \
  --net cifar10_LeNet \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 8 \
  --tolerable-fnr 0.01 \
  --upload-prefix "Results/stego_detector/$RUN_DIR/cv_k5"

