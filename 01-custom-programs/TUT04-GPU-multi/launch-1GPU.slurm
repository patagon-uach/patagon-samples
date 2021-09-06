#!/bin/bash
# IMPORTANT PARAMS
#SBATCH -p gpu                             # Submit to which partition
#SBATCH --gres=gpu:A100:1                  # GPU resources, format TYPE:device:quantity 

# OTHER PARAMS
#SBATCH -J random-field-IM3D	           # Name the job
#SBATCH -o random-field-IM3D-%j.out        # standard output to file
#SBATCH -e random-field-IM3D-%j.err        # standard error to file

# COMMANDS ON THE COMPUTE NODE
date                        # prints the date and time

# compile the GPU program
srun --container-name=cuda:11.4.1-devel nvidia-smi -L
srun --container-name=cuda:11.4.1-devel make
srun --container-name=cuda:11.4.1-devel echo "[CONTAINER] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# run the GPU program
srun --container-name=cuda:11.4.1-devel --pty \
./bin/trueke -l 256 32 -t 4.7 0.1 -a 120 2 500 10 -h 1.0 -s 5000 3000 100 5 1 1 -br 1 2000 -z 7919 -g 1
