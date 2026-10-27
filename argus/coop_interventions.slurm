#!/bin/bash
#SBATCH --job-name=coop_int
# NOTE: cluster-specific, replace with your SLURM account
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-coop-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-coop-%A_%a.err
module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

# ICBM CooP policy vs GP-uncertainty ordering (Table 9), CUB + CelebA, CLIP, 3 seeds, both models.
# task id -> (model, dataset, seed);  0-5 = VH-CBM, 6-11 = CBM-fair
i=$SLURM_ARRAY_TASK_ID
MODELS=(vh fair)
DATASETS=(cub celeba)
SEEDS=(132 213 312)
M=${MODELS[$((i / 6))]}
D=${DATASETS[$(((i % 6) / 3))]}
S=${SEEDS[$((i % 3))]}

echo "=== model=$M dataset=$D seed=$S ==="
if [ "$M" = "vh" ]; then
  # NOTE: cluster-specific path, replace with your own location
  LD=$(ls -d /leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/clip/${D}-510-activeours-RBF-${S}_*_SEED=${S})
  echo "load_dir=$LD"
  python eval_interventions_coop.py -dataset $D -backbone clip -seed $S -model vh -load_dir "$LD"
else
  python eval_interventions_coop.py -dataset $D -backbone clip -seed $S -model fair
fi
