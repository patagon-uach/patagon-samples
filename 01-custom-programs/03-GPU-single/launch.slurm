#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p L40              
#SBATCH --gpus=1            

# OTHER PARAMS
#SBATCH -J TUT03-single-GPU	         
#SBATCH -o log/TUT03-single-GPU-%j.out   
#SBATCH -e log/TUT03-single-GPU-%j.err   

# COMMANDS ON THE COMPUTE NODE
pwd                         # prints current working directory
date                        # prints the date and time

if [ "$#" -ne 3 ]; then
    echo "run as: sbatch launch.slurm <dev> <n> <mode>"
    echo "modes: 0:global-mem, 1:shared-mem"
    exit
fi
# compile the GPU program
srun --container-workdir=${PWD} --container-name=cuda-12 make
# run the GPU program
srun --container-workdir=${PWD} --container-name=cuda-12 ./prog $1 $2 $3
