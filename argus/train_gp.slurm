#!/bin/bash
#SBATCH --job-name=argo_gp
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --array=0-12%6           # Adjust based on #combinations and concurrency
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate


# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime

# Experiment combinations
seeds = [111,222,333,444,123,132,213,231,312,321,412,413]
datasets = ['dermamnist']
acquisition_functions = ['activeours']
kernels = ['RBF']

combinations = list(itertools.product(seeds, datasets, acquisition_functions, kernels))

# Get combination for this SLURM array task
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
seed, dataset, acq, kernel = combinations[idx]

# Set env vars for the experiment
env = os.environ.copy()
env['SEED'] = str(seed)
env['DS_NAME'] = dataset
env['ACQ_FN'] = acq
env['KERNEL'] = kernel
env['MODEL_NAME'] = 'dinov3'
env['VLM'] = 'dinov3'

# Timestamp for output filename
time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
outfile = f"{os.environ['SCRATCH']}/results/GPs/{time_date}_{seed}_{dataset}_{acq}_{kernel}.txt"

# Run the experiment
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    result = subprocess.run(["python", "OMCours.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    if result.stderr:
        f.write("\nERRORS:\n")
        f.write(result.stderr)
END
