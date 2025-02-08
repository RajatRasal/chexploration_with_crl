#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=invariant
#SBATCH --nodelist=mira10
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

cd /vol/biomedic3/rrr2417/chexploration_with_crl
source ../.bashrc

poetry run chexpert_train --nsamples 5 --invariant_sampling