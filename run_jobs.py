"""Script for running jobs on server"""

import os
import subprocess


def makejob(commit_id_param, crop_param):
    """Makes a new job"""
    return f"""#!/bin/bash

#SBATCH --job-name=cartoongan-fixed-crop-{crop_param}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=10:00:00
#SBATCH --output=../logslurms/slurm-%A_%a.out
#SBATCH --error=../logslurms/slurm-%A_%a.err

current_dir=`pwd`

echo "Session " crop_{crop_param}_${{SLURM_ARRAY_JOB_ID}}

echo "Copying the source directory and data"
date
mkdir $TMPDIR/cartoongan
rsync -r . $TMPDIR/cartoongan/

echo "Checking out the correct version of the code commit_id {commit_id_param}"
cd $TMPDIR/cartoongan/
git checkout {commit_id_param}


echo "Setting up the virtual environment"
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate

echo "Installing environment"
make install
pip install protobuf==3.9.2

echo "Training"
python3 -m src.run_train --gen-path weights/pretrained/trained_netG.pth --disc-path weights/pretrained/trained_netD.pth --crop-mode {crop_param}

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    """To launch a job"""
    with open("job.sbatch", "w", encoding="utf-8") as file:
        file.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
result = int(
    # pylint: disable=W1510
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

COMMIT_ID = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()


# Ensure the log directory exists
os.system("mkdir -p ~/logslurms")


# Launch the batch jobs

# lr_list = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
# for lr in lr_list:
#     submit_job(makejob(COMMIT_ID, lr))

# bs_list = [8, 12, 16, 20, 24, 32]
# for bs in bs_list:
#     submit_job(makejob(COMMIT_ID, bs))

crop_list = ["Resize", "Center", "Random"]
for crop in crop_list:
    submit_job(makejob(COMMIT_ID, crop))
