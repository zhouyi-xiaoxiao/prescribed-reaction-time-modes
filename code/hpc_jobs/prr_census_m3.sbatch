#!/bin/bash
#SBATCH --job-name=prr_census_m3
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# HPC item D: extreme-scale FK census, cell m3_eps0.05 (tag 98); stops cleanly at the deadline.
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_census.py simulate --cell m3_eps0.05 --paths 2e9 --workers 140 --deadline-min 215
echo "[job] end $(date -u +%FT%TZ)"
