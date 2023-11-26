#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=8         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --partition=amdgpufast      # partition name
#SBATCH --gres=gpu:1                 # 1 gpu per node
#SBATCH --mem=20G
#SBATCH --error=log/ST.err            # standard error file
#SBATCH --output=log/ST.out           # standard output file
#SBATCH --array 0-31%8

ml PyTorch3D/0.7.3-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0
#ml LieTorch/20210829-foss-2022a-CUDA-11.7.0

# TADY ZMENIT CD NA CESTU KDE MAS REPO A EXPORTOVAT PYTHONPATH!!!
cd $HOME/4D-RNSFP/
export PYTHONPATH=$PYTHONPATH:~/4D-RNSFP/

#python evaluate_flow.py ${SLURM_ARRAY_TASK_ID}
#python dev.py ${SLURM_ARRAY_TASK_ID}
python kitti_dev.py ${SLURM_ARRAY_TASK_ID}
