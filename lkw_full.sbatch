#!/bin/bash
# Full-data (no subsample) LKW posterior for the censuses the sensitivity sweep
# didn't already cover: 1, 2, 3, 4, 8 (5/6/7 exist as {c}_full.p). Together the
# eight {census}_full.p files are merged into cache/posterior_lkw_full.p by
# lkw_full_merge.py.
#
# Submit:  sbatch lkw_full.sbatch
#SBATCH --job-name=lkw_full
#SBATCH --partition=batch
#SBATCH --array=0-4%4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/pds/pds2/eddie/Dropbox/Research/forests2/py
#SBATCH --output=cache/slurm_logs/%A_%a.out

source /pds/pds2/eddie/anaconda3/etc/profile.d/conda.sh
conda activate forests3

CENSUSES=(1 2 3 4 8)
C=${CENSUSES[$SLURM_ARRAY_TASK_ID]}

python lkw_prior_job.py "$C" full
