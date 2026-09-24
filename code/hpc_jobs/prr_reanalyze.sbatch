#!/bin/bash
#SBATCH --job-name=prr_reanalyze
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
# Re-run the (deterministic) analyze steps of HPC items A/B after the small local-comparison inputs
# (N3/n3_summary.json, N5/n5_dt_ladder.json, robustness/dt_halving/m3_phase_boundary.json) were synced;
# the first analyses (jobs 5768126/5768127) ran without them. No new simulation.
source "${PRR_ARCHIVE_ROOT:?set PRR_ARCHIVE_ROOT to the archive root}/code/hpc_jobs/_common.sh"
"$PY" fb_hpc_n3full.py analyze
"$PY" fb_hpc_n5full.py analyze --replicas 2000
echo "[job] end $(date -u +%FT%TZ)"
