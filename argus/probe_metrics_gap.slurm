#!/bin/bash
#SBATCH --job-name=gap_metrics
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-gap-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-gap-%j.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# B1: SSCBM (knn + self-train) DIS @340   B2: CBM-frozen ResNet18 cub @340 ECCE-R + DIS
# Append-only + resumable, so resubmitting after a timeout loses nothing.
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg
python scripts/rebuttal_probe_metrics.py all
