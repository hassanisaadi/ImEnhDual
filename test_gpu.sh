#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=0-00:15
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.test_gpu.out
#SBATCH --job-name=test_gpu
echo "Starting run at: `date`"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/CUDA/intel2016.4/cuda8.0/cudnn/5.1/lib64
./test.lua -test_samples 1 -g -net_fname net/net_gpu_10.t7

