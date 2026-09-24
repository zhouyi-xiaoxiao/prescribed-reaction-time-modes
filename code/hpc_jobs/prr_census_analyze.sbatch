#!/bin/bash
#SBATCH --job-name=prr_census_an
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# HPC item D: census analysis after both census jobs (submit with --dependency=afterany:<m2>:<m3>).
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_census.py analyze
echo "[job] end $(date -u +%FT%TZ)"
