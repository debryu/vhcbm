#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --array=0-4%4            # Adjust based on #combinations and concurrency
#SBATCH --output=$FAST/results/baselines/slurm-%A_%a.out
#SBATCH --error=$FAST/results/baselines/slurm-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# Ensure results folder exists
mkdir -p $FAST/results/

# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime
try:
    datasets = ['cub','shapes3d','celeba','dermamnist']
    combinations = list(datasets)


    # Get combination for this SLURM array task
    idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
    dataset = combinations[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/baselines/baseline_logs_{time_date}_{dataset}.txt"

    # Set env vars for the experiment
    env = os.environ.copy()
    env['DS_NAME'] = dataset


    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        result = subprocess.run(["python", "-m", "scripts.baselines"], env=env, capture_output=True, text=True)
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)

except Exception as e:
    logfile = f"{os.environ['FAST']}/results/baselines/logs_{time_date}_{dataset}.txt"
    with open(logfile, "w") as f:
        f.write(str(e))

END