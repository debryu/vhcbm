#!/bin/bash
#SBATCH --job-name=rn18_cal
# NOTE: cluster-specific, replace with your SLURM account
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-rn18cal-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-rn18cal-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# B3: ECCE-R for VH-CBM over the ResNet18 backbone (cub, K=510, 3 seeds).
# Uses the *published* GP calibration procedure (scripts/calibration.py: 30 MC samples of the GP
# posterior, group softmax, per-column ECCE) so the number is comparable to the paper's table.
# These checkpoints are the resnet18 ones (date 2026_07_24_15_5*), NOT the CLIP cub runs --
# get_savename does not encode the backbone, so the date is what distinguishes them.
# NOTE: cluster-specific path, replace with your own location
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

SEEDS=(132 213 312)
PATHS=(
  "cub-510-activeours-RBF-132_2026_07_24_15_50_SEED=132.pt"
  "cub-510-activeours-RBF-213_2026_07_24_15_52_SEED=213.pt"
  "cub-510-activeours-RBF-312_2026_07_24_15_52_SEED=312.pt"
)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MP=${PATHS[$SLURM_ARRAY_TASK_ID]}

# NOTE: cluster-specific path, replace with your own location
OUTDIR=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/rn18_calibration
mkdir -p $OUTDIR

echo "=== seed $SEED  model_path $MP ==="
python scripts/calibration.py -dataset cub -seed $SEED \
    -model_path "$MP" \
    -save_dir "$OUTDIR/rn18_cub_SEED=$SEED"
