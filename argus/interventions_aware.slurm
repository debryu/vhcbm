#!/bin/bash
#SBATCH --job-name=int_aware
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-intaware-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-intaware-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# D1c: does intervention-aware training of the *task head only* remove the non-monotonic
# intervention curve? Retrains the final linear layer on the same saved concept means with random
# intervention augmentation. The GP concept layer, its annotation budget and all concept metrics
# are untouched, so this cannot change any other number in the paper.
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
export MPLBACKEND=agg

MP=(
  "cub-510-activeours-RBF-132_2026_04_05_11_56_SEED=132.pt"
  "cub-510-activeours-RBF-213_2026_04_05_11_56_SEED=213.pt"
  "cub-510-activeours-RBF-312_2026_04_05_11_56_SEED=312.pt"
)
echo "=== cub ${MP[$SLURM_ARRAY_TASK_ID]} ==="
python eval_interventions_aware.py -dataset cub -model_path "${MP[$SLURM_ARRAY_TASK_ID]}"
