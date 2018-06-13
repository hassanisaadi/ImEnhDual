#!/bin/bash
#SBATCH --account=def-shirani
#SBATCH --mem=5G
#SBATCH --time=0-00:15
#SBATCH --mail-type=ALL
#SBATCH --mail-user=graham.slurm.hassanih@gmail.com
#SBATCH --output=job_slurm/%J.test_cpu.out
#SBATCH --job-name=test_cpu_encdec
echo "Starting run at: `date`"
./test.lua -test_samples 1 -net_name enc_dec  -net_fname net/net_cpu_enc_dec_bs128_p32_bsps1_e10.t7 -data_dir ./data_mb2014_dark

