#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p gpu                       # Submit to which partition
#SBATCH --gres=gpu:A100:1            # GPU resources, format TYPE:device:quantity 

# OTHER PARAMS
#SBATCH -J TUT03-single-GPU	         # Name the job
#SBATCH -o TUT03-single-GPU-%j.out   # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e TUT03-single-GPU-%j.err   # Write the standard error to file named 'jMPItest-<job_number>.err'

# COMMANDS ON THE COMPUTE NODE
pwd                         # prints current working directory
date                        # prints the date and time

# compile the GPU program
srun --container-name=cuda-11.2.2 --container-image=cuda-11.2.2 make
# run the GPU program
srun --container-name=cuda-11.2.2 --container-image=cuda-11.2.2 ./prog 0 $((1024*30)) 1
