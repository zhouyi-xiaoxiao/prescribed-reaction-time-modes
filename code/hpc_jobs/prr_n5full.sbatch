#!/bin/bash
#SBATCH --job-name=prr_n5full
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
# HPC item A: N5 full dt ladder (tag 95), 3 cells x 3 seeds x 1e6 paths, exact OU + bridges.
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_n5full.py simulate --workers 140 --deadline-min 70
"$PY" fb_hpc_n5full.py analyze --replicas 2000
echo "[job] end $(date -u +%FT%TZ)"
