#!/bin/bash
#SBATCH --job-name=prr_validate
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
# Tiny end-to-end validation of fb_hpc_{census,headline,n5full,n3full} (throw-away output dir).
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
export PRR_HPC_WORK="$HOME/prr_hpc_test/work" PRR_HPC_OUT="$HOME/prr_hpc_test/out"
rm -rf "$HOME/prr_hpc_test"
mkdir -p "$PRR_HPC_WORK" "$PRR_HPC_OUT"
"$PY" "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/hpc_validate_tiny.py" census headline n5 n3 --workers 32
echo "[job] end $(date -u +%FT%TZ)"
