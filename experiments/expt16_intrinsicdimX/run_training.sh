#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-01:00
#SBATCH -p kempner
#SBATCH --mem=100g
#SBATCH -n 1 # if you use workers, this NEEDS to be the same number
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=jobsall.out
#SBATCH --error=errorlogs/job%a.err
#SBATCH --mail-type=END
#SBATCH --account=kempner_ba_lab
#SBATCH --array=1-5

# setup the environment
module load python/3.10.13-fasrc01
mamba activate spmax
# run training
python trainmodel.py --task_id $SLURM_ARRAY_TASK_ID
mamba deactivate