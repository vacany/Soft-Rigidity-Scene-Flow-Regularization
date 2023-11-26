#!/bin/bash
#SBATCH --partition amdgpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 12G
#SBATCH --gres gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output /home/vacekpa2/jupyter-notebook-current.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@login.rci.cvut.cz

Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: login.rci.cvut.cz
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"


ml PyTorch3D/0.7.3-foss-2022a-CUDA-11.7.0
ml matplotlib/3.5.2-foss-2022a
ml JupyterLab/3.5.0-GCCcore-11.3.0

ml LieTorch/20210829-foss-2022a-CUDA-11.7.0
ml faiss/1.7.3-foss-2022a-CUDA-11.7.0

cd $HOME/4D-RNSFP/

CUDA_LAUNCH_BLOCKING=1

jupyter-notebook --no-browser --port=${port} --ip=${node}
