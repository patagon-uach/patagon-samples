#!/bin/bash

#SBATCH -p cpu         	
#SBATCH -J TUT01-single-core    # Name the job
#SBATCH -o log/job-%j.out           # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e log/job-%j.err           # Write the standard error to file named 'jMPItest-<job_number>.err'
#SBATCH --nodes=1            	# Request N nodes
#SBATCH --ntasks-per-node=1 	# How many tas n cores or task per node
#SBATCH --cpus-per-task=1       # How many cores per task

pwd                         # prints current working directory
date                        # prints the date and time
make
srun ./prog $1      		# run the job
