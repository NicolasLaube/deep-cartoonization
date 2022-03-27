"""Script for running jobs on server"""

import os


def makejob(lr_param):
    """Makes a new job"""
    return f"""#!/bin/bash

#SBATCH --job-name=cartoongan-fixed-{lr_param}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=1:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-1

cd ~/cartoongan
python3 -m src.run_train --lr {lr_param}
"""


def submit_job(job):
    """To launch a job"""
    with open("job.sbatch", "w", encoding="utf-8") as file:
        file.write(job)
    os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")


# Launch the batch jobs
lr_list = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
for lr in lr_list:
    submit_job(makejob(lr))
