#!/bin/bash
#SBATCH --job-name=prr_headline
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# HPC item C: 1e7-walker x 2-seed direct-kill confirmation of 20 headline cells (tag 97).
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_headline.py simulate --workers 140 --deadline-min 40
"$PY" fb_hpc_headline.py analyze
echo "[job] end $(date -u +%FT%TZ)"
