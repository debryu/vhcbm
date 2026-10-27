#!/bin/bash
#SBATCH --job-name=rn_cbm
# NOTE: cluster-specific, replace with your SLURM account
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-rncbm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/ARGOCBM/slurm-rncbm-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# Task head + eval for the VH-CBM ResNet18 GPs (cub, K=510, seeds 132/213/312; Table 5).
# NOTE: cluster-specific path, replace with your own location
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus
python - <<'END'
import os, subprocess

MODELS = [
    "cub-510-activeours-RBF-132_2026_07_24_15_50_SEED=132.pt",
    "cub-510-activeours-RBF-213_2026_07_24_15_52_SEED=213.pt",
    "cub-510-activeours-RBF-312_2026_07_24_15_52_SEED=312.pt",
]
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
model = MODELS[idx]
seed = model.split("SEED=")[-1].replace(".pt", "")
save_dir = os.path.join(os.environ['FAST'], "results/ARGOCBM/resnet18")
os.makedirs(save_dir, exist_ok=True)

print("Training task head for:", model, flush=True)
r = subprocess.run(["python", "train_argocbm.py", "-dataset", "cub",
                    "-save_dir", save_dir, "-model_path", model, "-seed", seed])
if r.returncode != 0:
    raise SystemExit(r.returncode)

load_dir = os.path.join(save_dir, model.replace(".pt", ""))
print("Evaluating:", load_dir, flush=True)
r = subprocess.run(["python", "eval_argocbm.py", "-load_dir", load_dir])
raise SystemExit(r.returncode)
END
