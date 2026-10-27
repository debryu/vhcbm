#!/bin/bash
#SBATCH --job-name=randomc_dino
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-10%6
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/slurm-randomcdino-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/slurm-randomcdino-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus

# Jul 29 coverage fill for E4: task heads for the DINO randomc K=340 GPs, matching the seeds of
# the published DINO activeours heads (celeba/shapes3d/cub 111/222/333, dermamnist 222/333/444).
# Explicit list = every (dataset, seed) whose GP JSON has backbone=dinov3 AND whose checkpoint is
# staged (verified Jul 29).
# Jul 30: cub seed 333 (index 11) added — the lost 2026_03_29_16_54 ckpt was retrained by
# cub_randomc_dino_seed333.slurm (job 50716598); its K=340 round landed Jul 30 09:52 at
# f1_cal 0.8936 and is symlinked into models_staging under the NEW 2026_07_29_14_18 timestamp
# (size 6837896056, identical to the seed-111/222 cub dino ckpts).
MODELS=(
  "celeba-340-randomc-RBF-111_2026_03_27_11_07_SEED=111.pt"
  "celeba-340-randomc-RBF-222_2026_03_27_19_47_SEED=222.pt"
  "celeba-340-randomc-RBF-333_2026_03_28_04_33_SEED=333.pt"
  "cub-340-randomc-RBF-111_2026_03_29_16_28_SEED=111.pt"
  "cub-340-randomc-RBF-222_2026_03_29_16_29_SEED=222.pt"
  "dermamnist-340-randomc-RBF-222_2026_03_29_16_54_SEED=222.pt"
  "dermamnist-340-randomc-RBF-333_2026_03_29_17_08_SEED=333.pt"
  "dermamnist-340-randomc-RBF-444_2026_03_30_12_45_SEED=444.pt"
  "shapes3d-340-randomc-RBF-111_2026_03_27_11_07_SEED=111.pt"
  "shapes3d-340-randomc-RBF-222_2026_03_27_13_15_SEED=222.pt"
  "shapes3d-340-randomc-RBF-333_2026_03_27_14_36_SEED=333.pt"
  "cub-340-randomc-RBF-333_2026_07_29_14_18_SEED=333.pt"
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${MODEL##*SEED=}; SEED=${SEED%.pt}
DATASET=${MODEL%%-*}
SAVE_DIR=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/dino

echo "Training head for: $MODEL (dataset=$DATASET seed=$SEED)"
python train_argocbm.py -dataset "$DATASET" -save_dir "$SAVE_DIR" -model_path "$MODEL" -seed "$SEED" || exit 1
LOAD_DIR="$SAVE_DIR/${MODEL%.pt}"
echo "Evaluating: $LOAD_DIR"
python eval_argocbm.py -load_dir "$LOAD_DIR"
