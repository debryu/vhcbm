#!/bin/bash
#SBATCH --job-name=int_base
# NOTE: cluster-specific, replace with your SLURM account
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-intbase-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-intbase-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# D1b: concept interventions on the CBM-frozen baseline, identical protocol to VH-CBM.
# 4 datasets x 3 seeds, CLIP backbone. One array task per (dataset, seed) so a slow dataset
# cannot hold up the others; each appends its own JSON row.
# NOTE: cluster-specific path, replace with your own location
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

DATASETS=(shapes3d celeba cub dermamnist)
SEEDS=(132 213 312)
DS=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "=== $DS seed=$SEED ==="
python eval_interventions_baseline.py -dataset $DS -backbone clip -seed $SEED
