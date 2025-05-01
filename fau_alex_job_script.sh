#!/usr/bin/bash -l
#
# SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=lsenet_hpo
#SBATCH --output=res_hpo.txt

unset SLURM_EXPORT_ENV

module load python/3.9-anaconda
module load gcc/11.2.0
# source $WORK/python_virtual_envs/doubly_robust_env/bin/activate # activate the python virtual environment
conda activate clustering_env
python3 ./training.py >> output_training.txt
# ./cuda_application
