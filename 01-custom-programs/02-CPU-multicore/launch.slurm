#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p cpu     

# OTHER PARAMS
#SBATCH -J TUT02-CPU-openmp               # Name the job
#SBATCH -o log/TUT02-CPU-openmp-%j.out    # standard output file
#SBATCH -e log/TUT02-CPU-openmp-%j.err    # errors file
#SBATCH --nodes=1            	          # Request N nodes
#SBATCH --ntasks=1 	                      # Request n cores or task per node  
#SBATCH --cpus-per-task=8                 # CPU Cores per task
#SBATCH --mem=65536            	          # Request MB RAM for JOB

pwd                                         
date                                        

if [ "$#" -ne 2 ]; then
    echo "run as: sbatch launch.slurm <n> <nt>"
    exit
fi
# compile the program
srun  make
# run the program
srun ./prog $1 $2
