#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=10G
#SBATCH --time=0-01:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.test_cpu.out
#SBATCH --job-name=test_cpu
echo "Starting run at: `date`"
./test.lua -test_samples 1 -net_fname net/net_cpu_2.t7

