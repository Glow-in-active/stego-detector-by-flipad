#!/bin/bash
#SBATCH --job-name=stego_cache
#SBATCH --output=stego_cache-%j.log
#SBATCH --error=stego_cache-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
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

# ---- S3 creds ----
# ВСТАВЬ СЮДА КЛЮЧИ (на сервере это ок, но НЕ коммить)
export AWS_ACCESS_KEY_ID="ВАШ_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="ВАШ_SECRET_ACCESS_KEY"
export AWS_REGION="ru-central1"
export AWS_DEFAULT_REGION="ru-central1"
export ENDPOINT="https://storage.yandexcloud.net"
export BUCKET="stegopractice"

RUN_DIR="runs/cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python3 -u hpc/run_stego_vs_sd_dct.py cache \
  --out-dir "$RUN_DIR" \
  --size 32 \
  --workers 32 \
  --upload-prefix "Results/stego_detector/$RUN_DIR"

