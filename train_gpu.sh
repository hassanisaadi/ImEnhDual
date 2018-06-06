#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=0-02:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train_gpu.out
#SBATCH --job-name=train_gpu
echo "Starting run at: `date`"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/CUDA/intel2016.4/cuda8.0/cudnn/5.1/lib64
./train.lua -sceneNum 22 -g -bs 150 -patchSizeTr 32 -epoch 200

