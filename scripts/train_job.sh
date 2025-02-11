#!/bin/bash
#SBATCH -p gpus24,gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=invariant
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

cd /vol/biomedic3/rrr2417/chexploration_with_crl
source ../.bashrc

TRAIN_ARGS="""
    --nsamples $1 \
    --inv-loss-coefficient 1 \
    --dataset-train $2 \
    --dataset-test $3 \
    --protected-race-set-train $4 \
    --protected-race-set-test $5
"""

if [ "$6" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --invariant-sampling"
fi

if [ "$7" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --test"
fi

TRAIN_CMD="poetry run chexpert_train $TRAIN_ARGS"
echo $TRAIN_CMD

eval $TRAIN_CMD
