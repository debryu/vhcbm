#!/bin/bash
#SBATCH --job-name=s3d_activations
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=22:00:00
# NOTE: cluster-specific, replace with your log directory
#SBATCH --output=/leonardo_scratch/fast/IscrC_ARGO/logs/act-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_ARGO/logs/act-%j.err

module load python
module load cuda

# NOTE: cluster-specific, replace with your Python environment
source /leonardo/home/userexternal/ndebole0/projects/vlm/bin/activate

# NOTE: cluster-specific path, replace with your own location
python /leonardo/home/userexternal/ndebole0/projects/argus/argus/scripts/save_activations.py
