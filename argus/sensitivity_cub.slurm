#!/bin/bash
#SBATCH --job-name=sens_cub
#SBATCH --account=iscrc_argo
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-29%15
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-senscub-%A_%a.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/results/GPs/slurm-senscub-%A_%a.err

module load python
module load cuda
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate
cd /leonardo_scratch/fast/IscrC_ARGO/argobackup/argus

mkdir -p $SCRATCH/results/GPs/GPs $SCRATCH/models

# Jul 29 coverage fill for E2: cub — the dataset the rebuttal sweep skipped for cost (~20 h/run,
# cf. job 50173198). Full mirror of the celeba extension (user decision Jul 29): 5-point
# lengthscale sweep at outputscale 1.0, both arms, 3 seeds = 30 runs. ρ=0.1-fixed may hit the
# 24 h limit like celeba did at 8 h — JSONs are written per AL round, so partial runs are still
# reportable (mark † as in the celeba table).
python - <<'END'
import os, itertools, subprocess

configs = [(0.1, 1.0), (0.5, 1.0), (1.41, 1.0), (5.0, 1.0), (10.0, 1.0)]
learn_modes = ['1', '0']
datasets = ['cub']
seeds = [132, 213, 312]

combinations = list(itertools.product(configs, learn_modes, datasets, seeds))
assert len(combinations) == 30, len(combinations)

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
(ls, outs), learn, dataset, seed = combinations[idx]

env = os.environ.copy()
env['DS_NAME'] = dataset
env['SEED'] = str(seed)
env['ACQ_FN'] = 'activeours'
env['KERNEL'] = 'RBF'
env['VLM'] = 'ViT-L/14'
env['LENGTHSCALE'] = str(ls)
env['OUTPUTSCALE'] = str(outs)
env['LEARN_KERNEL'] = learn

print(f"=== cub seed={seed} LS={ls} OS={outs} LK={learn} ===", flush=True)
result = subprocess.run(["python", "OMCours.py"], env=env)
raise SystemExit(result.returncode)
END
