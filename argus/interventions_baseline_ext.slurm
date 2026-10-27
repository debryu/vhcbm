#!/bin/bash
#SBATCH --job-name=int_base_ext
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-intbext-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-intbext-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# Jul 29 coverage fill: CBM-frozen intervention baseline on the two backbones it was missing.
# Tasks 0-11: dinov3 x 4 datasets x 3 seeds; tasks 12-14: resnet18_cub x cub x 3 seeds.
# Same protocol/script as the CLIP rows already in D1_interventions_baseline.jsonl.
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

i=$SLURM_ARRAY_TASK_ID
SEEDS=(132 213 312)
if [ $i -lt 12 ]; then
  DATASETS=(shapes3d celeba cub dermamnist)
  DS=${DATASETS[$((i / 3))]}
  BB=dinov3
else
  DS=cub
  BB=resnet18_cub
fi
SEED=${SEEDS[$((i % 3))]}

echo "=== $DS backbone=$BB seed=$SEED ==="
python eval_interventions_baseline.py -dataset $DS -backbone $BB -seed $SEED
