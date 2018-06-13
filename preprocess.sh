#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=50G
#SBATCH --time=0-01:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.preprocess.out
module load python/2.7.14
module load gcc/5.4.0 opencv/2.4.13.3
source /home/hassanih/ENV/bin/activate
./preprocess.py
