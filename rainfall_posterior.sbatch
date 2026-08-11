#!/bin/bash
# Slurm array over all (census_year, kernel_duration) pairs for the rainfall
# posterior in 'BCI rainfall 2024-02-21.ipynb'. 7 census years x 13 kernel
# durations = 91 tasks; each runs in its own process so memory is returned to
# the OS when the task ends.
#
# Submit from ./py:   sbatch rainfall_posterior.sbatch
# Merge when done:    python rainfall_posterior_merge.py
#SBATCH --job-name=rain_post
#SBATCH --partition=batch
#SBATCH --array=0-90%4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/pds/pds2/eddie/Dropbox/Research/forests2/py
#SBATCH --output=cache/slurm_logs/%A_%a.out

source /pds/pds2/eddie/anaconda3/etc/profile.d/conda.sh
conda activate forests3

CENSUS_YEARS=(1990 1995 2000 2005 2010 2015 2020)
CY=${CENSUS_YEARS[$((SLURM_ARRAY_TASK_ID / 13))]}
KD=$((2 + SLURM_ARRAY_TASK_ID % 13))

echo "host=$(hostname) census_year=$CY kernel_duration=$KD"
python rainfall_posterior_job.py $CY $KD
