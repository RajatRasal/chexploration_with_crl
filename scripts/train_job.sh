#!/bin/bash
#SBATCH -p gpus24,gpus48
#SBATCH --gres gpu:1
#SBATCH --job-name=invariant
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

# cd /vol/biomedic3/rrr2417/chexploration_with_crl
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

TRAIN_ARGS="""
    --nsamples $1 \
    --inv-loss-coefficient 1 \
    --dataset-train $2 \
    --dataset-test $3 \
    --protected-race-set-train $4 \
    --protected-race-set-test $5 \
    --seed $6 \
    --model-type $7
"""

if [ "$8" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --invariant-sampling"
fi

if [ "$9" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --test"
fi

TRAIN_CMD="poetry run race_invariance $TRAIN_ARGS"
echo $TRAIN_CMD

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo Train job failed
    exit 1
fi
