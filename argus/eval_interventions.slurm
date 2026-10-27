#!/bin/bash
#SBATCH --job-name=argo_interv
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-26%8
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-interv-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM/slurm-interv-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# Test-time interventions (Table 9) on all clip+dino VH-CBM runs plus the 3 resnet18 runs.
# group_scores_ucp uses float64 with eps=1e-12 to avoid NaNs in the entropy ordering.
python - <<'END'
import os, subprocess

# NOTE: cluster-specific path, replace with your own location
base = '/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM'
dirs = []
for sub in ['clip', 'dino', 'resnet18']:
    p = os.path.join(base, sub)
    for d in sorted(os.listdir(p)):
        full = os.path.join(p, d)
        if os.path.isdir(full) and '-510-' in d and os.path.exists(os.path.join(full, 'W_g.pt')):
            dirs.append(full)
assert len(dirs) == 27, f"expected 27 run dirs, found {len(dirs)}"

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
load_dir = dirs[idx]
print("Intervening on:", load_dir)
result = subprocess.run(["python", "eval_interventions.py", "-load_dir", load_dir])
raise SystemExit(result.returncode)
END
