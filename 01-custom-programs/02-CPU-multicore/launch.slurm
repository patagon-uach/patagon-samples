#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p cpu                                	                    # Submit to which partition

# OTHER PARAMS
#SBATCH -J TUT02-CPU-openmp                 # Name the job
#SBATCH -o TUT02-CPU-openmp-%j.out          # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e TUT02-CPU-openmp-%j.err          # Write the standard error to file named 'jMPItest-<job_number>.err'
#SBATCH --nodes=1            	            # Request N nodes
#SBATCH --ntasks=1 	                        # Request n cores or task per node  
#SBATCH --cpus-per-task=32                  # CPU Cores per task
#SBATCH --mem=65536            	            # Request MB RAM for JOB

pwd                                         # prints current working directory
date                                        # prints the date and time

# compile the program
srun  make
# run the program
srun ./prog $((2**16)) 32
