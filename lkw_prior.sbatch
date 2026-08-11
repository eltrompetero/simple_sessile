#!/bin/bash
# Slurm array over the LKW prior/procedure sensitivity variants for censuses
# 5, 6, 7 (see lkw_prior_job.py). 3 censuses x 7 variants = tasks 0-20 in
# census-major order; task 21 is the census-6 deeptree extra.
#
# Submit one task first (e.g. 6_seedA):  sbatch --array=7 lkw_prior.sbatch
# Then the rest:                         sbatch --array=0-6,8-21%4 lkw_prior.sbatch
#SBATCH --job-name=lkw_prior
#SBATCH --partition=batch
#SBATCH --array=0-21%4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/pds/pds2/eddie/Dropbox/Research/forests2/py
#SBATCH --output=cache/slurm_logs/%A_%a.out

source /pds/pds2/eddie/anaconda3/etc/profile.d/conda.sh
conda activate forests3

CENSUSES=(5 6 7)
VARIANTS=(seedA seedB seedC full kprior1 kprior2 rprior)
if [ "$SLURM_ARRAY_TASK_ID" -eq 21 ]; then
    C=6
    V=deeptree
else
    C=${CENSUSES[$((SLURM_ARRAY_TASK_ID / 7))]}
    V=${VARIANTS[$((SLURM_ARRAY_TASK_ID % 7))]}
fi

echo "host=$(hostname) census=$C variant=$V"
python lkw_prior_job.py $C $V
