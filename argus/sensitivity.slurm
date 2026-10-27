#!/bin/bash
#SBATCH --job-name=argo_sens
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --array=0-83%6
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sens-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sens-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

mkdir -p $SCRATCH/results/GPs/GPs $SCRATCH/models

# E2 rebuttal sweep: GP kernel hyperparameter sensitivity (lengthscale/outputscale),
# learned vs frozen, on dermamnist + shapes3d, CLIP backbone, activeours acquisition.
python - <<'END'
import os, itertools, subprocess, datetime

# (lengthscale, outputscale): lengthscale sweep at OS=1.0, plus outputscale sweep at LS=1.41
configs = [(0.1, 1.0), (0.5, 1.0), (1.41, 1.0), (5.0, 1.0), (10.0, 1.0),
           (1.41, 0.1), (1.41, 10.0)]
learn_modes = ['1', '0']
datasets = ['dermamnist', 'shapes3d']
seeds = [132, 213, 312]

combinations = list(itertools.product(configs, learn_modes, datasets, seeds))
assert len(combinations) == 84

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
outfile = f"{os.environ['SCRATCH']}/results/GPs/sens_{time_date}_{seed}_{dataset}_LS{ls}_OS{outs}_LK{learn}.txt"
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    result = subprocess.run(["python", "OMCours.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)
END
