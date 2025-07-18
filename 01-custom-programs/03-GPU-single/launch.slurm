#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p L40                       # Submit to which partition
#SBATCH --gpus=1            # GPU resources, format TYPE:device:quantity 

# OTHER PARAMS
#SBATCH -J TUT03-single-GPU	         # Name the job
#SBATCH -o TUT03-single-GPU-%j.out   # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e TUT03-single-GPU-%j.err   # Write the standard error to file named 'jMPItest-<job_number>.err'

# COMMANDS ON THE COMPUTE NODE
pwd                         # prints current working directory
date                        # prints the date and time

# compile the GPU program
srun --container-workdir=${PWD} --container-name=cuda-12 make
# run the GPU program
srun --container-workdir=${PWD} --container-name=cuda-12 ./prog 0 $((1024*40)) 1
