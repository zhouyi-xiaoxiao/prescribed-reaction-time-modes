#!/bin/bash
#SBATCH --job-name=prr_n3full
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
# HPC item B: N3 contact/field factorial, GPT-6 full spec (tag 96), 2 anchors x 3 seeds x 1e6, exact OU ladder.
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_n3full.py simulate --workers 140 --deadline-min 70
"$PY" fb_hpc_n3full.py analyze
echo "[job] end $(date -u +%FT%TZ)"
