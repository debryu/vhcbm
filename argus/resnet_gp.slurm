#!/bin/bash
#SBATCH --job-name=argo_rngp
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-rngp-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-rngp-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# E1 symmetric argument: VH-CBM GP concept layer over frozen ResNet18 features (cub only).
# Completes the triple: VH-CBM-ResNet18 vs FairCBM-ResNet18 (done) vs paper's CBM-ResNet18.
python - <<'END'
import os, subprocess, datetime

seeds = [132, 213, 312]
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
seed = seeds[idx]

env = os.environ.copy()
env['SEED'] = str(seed)
env['DS_NAME'] = 'cub'
env['ACQ_FN'] = 'activeours'
env['KERNEL'] = 'RBF'
env['VLM'] = 'resnet18_cub'
env['MODEL_NAME'] = 'resnet18_cub'

time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
outfile = f"{os.environ['SCRATCH']}/results/GPs/rngp_{time_date}_{seed}_cub_activeours_RBF.txt"

os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    result = subprocess.run(["python", "OMCours.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)
raise SystemExit(result.returncode)
END
