#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=50G
#SBATCH --time=0-00:30
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.train_cpu_smplcnn.out
#SBATCH --job-name=train_cpu_smplcnn
echo "Starting run at: `date`"
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/CUDA/intel2016.4/cuda8.0/cudnn/5.1/lib64
./train.lua -sceneNum 21 -arch simple_cnn -epoch 10 -patchSizeTr 64 -bs 128 -bsperscene 1 -data_dir ./data_mb2014_dark

