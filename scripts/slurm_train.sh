#!/bin/bash
#SBATCH -p gpus24
#SBATCH --gres gpu:1
#SBATCH --job-name=chexpert_train
#SBATCH --nodelist=mira09
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

cd /vol/biomedic3/rrr2417/chexploration_with_crl
source ../.bashrc

poetry run chexpert_train