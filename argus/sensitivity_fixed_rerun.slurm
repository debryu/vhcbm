#!/bin/bash
#SBATCH --job-name=sens_fix
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-41%14
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sensfix-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sensfix-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

mkdir -p $SCRATCH/results/GPs/GPs $SCRATCH/models

# E2 RERUN of the FIXED arm only (LEARN_KERNEL=0), after fixing the freeze bug.
#
# Why: the original sweep's "fixed" arm never froze anything. gpytorch re-enables requires_grad on
# raw_lengthscale/raw_outputscale during the forward pass, so the requires_grad=False set in
# reset_models() was undone on the first forward and Adam trained the kernel anyway -- verified both
# in the saved JSONs (10/10 latents drifted in all 28 configs x 3 seeds) and by a direct probe
# (requires_grad flips False -> True across m(X)). Fixed by excluding those tensors from the
# optimizer's param groups (MCSVGPDO.train), which the forward pass cannot undo.
#
# The learned arm (LEARN_KERNEL=1) is unaffected by the bug and is NOT rerun: with nothing frozen,
# frozen_ids is empty and the code path is byte-identical to before.
#
# Savename tag stays LS{ls}_OS{os}_LK{learn}; the new files are distinguished by date, so filter on
# date when aggregating (the old invalid LK0 files are still on disk).
python - <<'END'
import os, itertools, subprocess, datetime

configs = [(0.1, 1.0), (0.5, 1.0), (1.41, 1.0), (5.0, 1.0), (10.0, 1.0),
           (1.41, 0.1), (1.41, 10.0)]
learn_modes = ['0']                      # fixed arm ONLY
datasets = ['dermamnist', 'shapes3d']
seeds = [132, 213, 312]

combinations = list(itertools.product(configs, learn_modes, datasets, seeds))
assert len(combinations) == 42, len(combinations)

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
           f"sensfix_{time_date}_{seed}_{dataset}_LS{ls}_OS{outs}_LK{learn}.txt")
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    result = subprocess.run(["python", "OMCours.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)
END
