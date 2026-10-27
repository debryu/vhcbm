#!/bin/bash
#SBATCH --job-name=sens_cel
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --array=0-29%15
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-senscel-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-senscel-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

mkdir -p $SCRATCH/results/GPs/GPs $SCRATCH/models

# E2 EXTENSION to a second real-world dataset: celeba (39 concepts).
# Closes the "you only swept dermamnist and shapes3d" attack surface.
#
# Scope deliberately reduced to the 5-point LENGTHSCALE sweep at outputscale=1.0, both arms,
# 3 seeds = 30 runs. The outputscale-only points are already covered on dermamnist+shapes3d.
# cub is NOT swept: its GPs cost ~20 h per run (cf. job 50173198, 20:46:58 for the ResNet18 cub GPs),
# so 42 cub runs cannot fit in the rebuttal window. Say so explicitly rather than implying coverage.
#
# Best-effort: this is a nice-to-have, not a gate. If it will not finish before the deadline,
# cancel it and report the dermamnist/shapes3d sweep only.
python - <<'END'
import os, itertools, subprocess, datetime

configs = [(0.1, 1.0), (0.5, 1.0), (1.41, 1.0), (5.0, 1.0), (10.0, 1.0)]
learn_modes = ['1', '0']
datasets = ['celeba']
seeds = [132, 213, 312]

combinations = list(itertools.product(configs, learn_modes, datasets, seeds))
assert len(combinations) == 30, len(combinations)

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
(ls, outs), learn, dataset, seed = combinations[idx]

env = os.environ.copy()
env['SEED'] = str(seed)
env['DS_NAME'] = dataset
env['ACQ_FN'] = 'activeours'
env['KERNEL'] = 'RBF'
env['VLM'] = 'ViT-L/14'
env['MODEL_NAME'] = 'ViT-L/14'
env['LENGTHSCALE'] = str(ls)
env['OUTPUTSCALE'] = str(outs)
env['LEARN_KERNEL'] = learn

time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
outfile = (f"{os.environ['SCRATCH']}/results/GPs/"
           f"senscel_{time_date}_{seed}_{dataset}_LS{ls}_OS{outs}_LK{learn}.txt")
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    result = subprocess.run(["python", "OMCours.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)
END
