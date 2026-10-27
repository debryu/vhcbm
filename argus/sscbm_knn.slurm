#!/bin/bash
#SBATCH --job-name=sscbm_knn
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-23%8
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/SSCBM/slurm-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# E5 rebuttal (upgraded to P1): SSCBM-style pseudo-labeling over frozen embeddings.
# 24 = 4 datasets x {ViT-L/14, dinov3} x seeds {132,213,312}.
python - <<'END'
import os, itertools, subprocess

datasets = ['shapes3d', 'dermamnist', 'celeba', 'cub']
backbones = ['ViT-L/14', 'dinov3']
seeds = [132, 213, 312]

combinations = list(itertools.product(datasets, backbones, seeds))
assert len(combinations) == 24

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
dataset, backbone, seed = combinations[idx]

env = os.environ.copy()
env['DS_NAME'] = dataset
env['VLM'] = backbone
env['SEED'] = str(seed)
env['DEVICE'] = 'cuda'

result = subprocess.run(["python", "scripts/sscbm_knn.py"], env=env)
raise SystemExit(result.returncode)
END
