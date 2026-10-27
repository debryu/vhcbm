#!/bin/bash
#SBATCH --job-name=argo_randomc
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --array=0-11%6
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/slurm-randomc-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/slurm-randomc-%A_%a.err

module load python
module load cuda
# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# E4 rebuttal: train + eval task heads for the randomc K=340 GPs (seeds 132/213/312, CLIP),
# giving the task-level randomc-vs-activeours acquisition comparison.
python - <<'END'
import os, json, subprocess

BASE = os.environ['FAST']
GPs = os.path.join(BASE, 'results/GPs/GPs')
models = os.listdir(os.path.join(BASE, "models_staging"))
SEEDS = ['132', '213', '312']

to_run = []
for s in os.listdir(GPs):
    if 'MCSVGP' not in s: continue
    try:
        args = json.load(open(os.path.join(GPs, s)))[0]
    except Exception: continue
    if args.get('acq_fn') != 'randomc' or args.get('backbone') != 'ViT-L/14': continue
    seed = s.split("SEED=")[1].replace('.json', '')
    if seed not in SEEDS: continue
    to_run.append((seed, s.split("-")[0], s.split("-")[1].split("_")[1]))

selected = set()
for (seed, date, dataset) in to_run:
    for m in models:
        if seed in m and date in m and dataset in m and 'randomc' in m and m.split('-')[1] == '340':
            selected.add(m)
selected = sorted(selected)
assert len(selected) == 12, f"expected 12 randomc models, found {len(selected)}: {selected}"

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
model = selected[idx]
seed = model.split("SEED=")[-1].replace(".pt", "")
dataset = model.split("-")[0]
save_dir = os.path.join(BASE, "results/ARGOCBM/clip")

print("Training head for:", model)
r = subprocess.run(["python", "train_argocbm.py",
                    "-dataset", dataset,
                    "-save_dir", save_dir,
                    "-model_path", model,
                    "-seed", seed])
if r.returncode != 0:
    raise SystemExit(r.returncode)

load_dir = os.path.join(save_dir, model.replace(".pt", ""))
print("Evaluating:", load_dir)
r = subprocess.run(["python", "eval_argocbm.py", "-load_dir", load_dir])
raise SystemExit(r.returncode)
END
