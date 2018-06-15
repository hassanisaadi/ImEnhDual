#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=5G
#SBATCH --time=0-00:05
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train_cpu.out
#SBATCH --job-name=train_cpu
echo "Starting run at: `date`"
./train.lua \
  -bs 16\
  -patchSizeTr 32\
  -lr 0.001\
  -sceneNum 2\
  -beta1 0.9\
  -beta2 0.999\
  -data_dir ./data_mb2014_dark\
  -arch enc_dec\
  -epoch_start 6\
  -epoch_end 10\
  -continueLearning\
  -last_net_name ./net/net_cpu_enc_dec_bs16_p32_e5.t7

