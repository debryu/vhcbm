#!/bin/bash
#SBATCH --job-name=argo_gp
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --array=0-12%6          # Adjust based on #combinations and concurrency
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/argocbmEvalslurm-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/argocbmEval-slurm-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate


# Map SLURM_ARRAY_TASK_ID to your experiment combination
python - <<'END'
import os, itertools, subprocess, datetime

DATASET = 'shapes3d'
FAST_STORAGE = os.environ["SCRATCH"]
to_evaluate = []
# Experiment combinations
BASE = f"{os.environ['SCRATCH']}/results/ARGOCBM/dinov3"
models = os.listdir(BASE)
for folder in models:
    #print(folder)
    if "-340-" not in folder and "-510-" not in folder:
        continue
    #    print(folder)
    if DATASET not in folder:
        continue
    
    metrics_path = os.path.join(BASE,folder,"metrics.txt")
    to_evaluate.append(os.path.join(BASE,folder))
    #if os.path.exists(metrics_path):
    #    continue
    #else:
    #    if os.path.exists(os.path.join(FAST_STORAGE,"results/ARGOCBM",folder,"args.txt")): 
    #        to_evaluate.append(os.path.join(FAST_STORAGE,"results/ARGOCBM",folder))

# Get combination for this SLURM array task
idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
if idx < len(to_evaluate):
    load_dir = to_evaluate[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/CBMs/{time_date}_evalARGOCBMeval.txt"
    env = os.environ.copy()
    print(load_dir)
    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        result = subprocess.run(["python", "eval_argocbm.py", 
                                "-load_dir", load_dir,
                                ], env=env, capture_output=True, text=True)
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)
END
