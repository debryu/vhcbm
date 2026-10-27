#!/bin/bash
#SBATCH --job-name=coop_ext
# NOTE: cluster-specific, replace with your SLURM account
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-coopext-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-coopext-%A_%a.err
module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

# Jul 29 coverage fill for D6 (ICBM CooP policy): the settings the original 12-task array skipped.
#   0-5   : shapes3d CLIP        (vh seeds 132/213/312, fair seeds 132/213/312)
#   6-23  : DINO cub/celeba/shapes3d (vh seeds 111/222/333 = the dino run seeds;
#           fair seeds 132/213/312 = the FAIRCBM/dinov3 seeds — seed sets differ by design,
#           they are independent models; noted in the jsonl rows via the seed field)
# dermamnist stays excluded (1 concept group -> no ordering to compare).
i=$SLURM_ARRAY_TASK_ID

if [ $i -lt 6 ]; then
  BB=clip; D=shapes3d
  M=$([ $((i / 3)) -eq 0 ] && echo vh || echo fair)
  SEEDS=(132 213 312)
  S=${SEEDS[$((i % 3))]}
else
  BB=dinov3
  j=$((i - 6))
  DATASETS=(cub celeba shapes3d)
  D=${DATASETS[$((j / 6))]}
  M=$([ $(((j % 6) / 3)) -eq 0 ] && echo vh || echo fair)
  if [ "$M" = "vh" ]; then SEEDS=(111 222 333); else SEEDS=(132 213 312); fi
  S=${SEEDS[$((j % 3))]}
fi

echo "=== model=$M dataset=$D backbone=$BB seed=$S ==="
if [ "$M" = "vh" ]; then
  SUB=$([ "$BB" = "clip" ] && echo clip || echo dino)
  # NOTE: cluster-specific path, replace with your own location
  LD=$(ls -d /leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/${SUB}/${D}-510-activeours-RBF-${S}_*_SEED=${S})
  echo "load_dir=$LD"
  python eval_interventions_coop.py -dataset $D -backbone $BB -seed $S -model vh -load_dir "$LD"
else
  python eval_interventions_coop.py -dataset $D -backbone $BB -seed $S -model fair
fi
