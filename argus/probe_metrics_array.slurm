#!/bin/bash
#SBATCH --job-name=gap_arr
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-50%26
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-gaparr-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-gaparr-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# B1/B2 as a SLURM array (one row per task). The serial version needed ~10 min/row for celeba and
# far more for cub (DCI fits a GBT per code-factor pair: 112x112 for cub), which does not fit the
# window -- the reference FairCBM metrics job parallelised for exactly this reason.
# Each task writes its own JSON part file, so no two tasks race on a shared jsonl.
# Task order: SSCBM-KNN DIS (0-23) -> ResNet18 CBM-frozen ECCE-R+DIS (24-26) -> self-train (27-50),
# so the rows the tables need are computed first.
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg
export OMP_NUM_THREADS=8

python scripts/rebuttal_probe_metrics.py index $SLURM_ARRAY_TASK_ID
