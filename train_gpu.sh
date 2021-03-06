#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=0-00:10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train_gpu.out
#SBATCH --job-name=train_gpu
echo "Starting run at: `date`"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/CUDA/intel2016.4/cuda8.0/cudnn/5.1/lib64
./train.lua\
  -g\
  -bs 128\
  -patchSizeTr 32\
  -lr 0.001\
  -sceneNum 21\
  -beta1 0.9\
  -beta2 0.999\
  -data_dir ./data_mb2014_dark\
  -arch enc_dec\
  -epoch_start 1\
  -epoch_end 10
# -continueLearning\
# -last_net_name ./net/net_cpu_...

