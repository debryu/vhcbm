#!/bin/bash
#SBATCH --job-name=sens_dino
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-83%12
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sensdino-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-sensdino-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus

mkdir -p $SCRATCH/results/GPs/GPs $SCRATCH/models

# Jul 29 coverage fill for E2: the kernel sweep on the DINO backbone (the published sweep is
# CLIP-only). Same 7-config grid as the original CLIP sweep, dermamnist + shapes3d (the cheap
# datasets), both arms, 3 seeds = 84 runs. JSONs carry a `_dinov3` suffix in the savename tag
# (OMCours.py appends it for non-CLIP backbones) so they cannot collide with the CLIP JSONs.
python - <<'END'
import os, itertools, subprocess

configs = [(0.1, 1.0), (0.5, 1.0), (1.41, 1.0), (5.0, 1.0), (10.0, 1.0), (1.41, 0.1), (1.41, 10.0)]
learn_modes = ['1', '0']
datasets = ['dermamnist', 'shapes3d']
seeds = [132, 213, 312]

combinations = list(itertools.product(configs, learn_modes, datasets, seeds))
assert len(combinations) == 84, len(combinations)

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
(ls, outs), learn, dataset, seed = combinations[idx]

env = os.environ.copy()
env['DS_NAME'] = dataset
env['SEED'] = str(seed)
env['ACQ_FN'] = 'activeours'
env['KERNEL'] = 'RBF'
env['VLM'] = 'dinov3'
env['LENGTHSCALE'] = str(ls)
env['OUTPUTSCALE'] = str(outs)
env['LEARN_KERNEL'] = learn

print(f"=== dinov3 {dataset} seed={seed} LS={ls} OS={outs} LK={learn} ===", flush=True)
result = subprocess.run(["python", "OMCours.py"], env=env)
raise SystemExit(result.returncode)
END
