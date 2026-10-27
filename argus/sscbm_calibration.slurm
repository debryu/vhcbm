#!/bin/bash
#SBATCH --job-name=sscbm_cal
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-cal-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-cal-%j.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# E5 calibration: ECCE-R (paper's ECCE), ECCE-MAD, ECE for the 48 SSCBM N=340 runs.
# Appends per run to rebuttal_drafts/sscbm_calibration_N340.jsonl and skips completed runs,
# so it is safe to resubmit.
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg
python scripts/sscbm_calibration.py
