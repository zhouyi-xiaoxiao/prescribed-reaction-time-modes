# Sourced by the prr_*.sbatch scripts (Isambard 3, partition grace).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PRR_HPC_WORK="$HOME/prr_hpc_work"
PY="${PRR_PYTHON:-python3}"
CODE="${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code"
cd "$CODE"
echo "[job] $SLURM_JOB_NAME $SLURM_JOB_ID on $(hostname) cpus=$SLURM_CPUS_PER_TASK start $(date -u +%FT%TZ)"
"$PY" -c "import numpy, sys; print('[job] python', sys.version.split()[0], 'numpy', numpy.__version__)"
