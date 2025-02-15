#!/bin/bash
#SBATCH -p gpus48
#SBATCH --nodelist=mira01,loki 
#SBATCH --gres=gpu:1
#SBATCH --job-name=breast_invariant
#SBATCH --nodes=1
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

# Source Virtual environment (conda)
if [ "$(whoami)" == "agk21" ]; then
    . "/vol/biomedic3/agk21/anaconda3/etc/profile.d/conda.sh"
    echo "Conda environment sourced successfully."
    conda activate chexploration
    echo "Conda environment activated successfully."
else
    export PATH="/vol/biomedic3/rrr2417/.local/bin:$PATH"
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
fi

TRAIN_ARGS="""
    --nsamples $1 \
    --inv-loss-coefficient 1 \
    --dataset-train $2 \
    --dataset-test $3 \
    --view-set-train $4 \
    --view-set-test $5 \
    --seed $6 \
    --model-type $7 \
    --epochs ${10}
"""

if [ "$8" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --invariant-sampling"
fi

if [ "$9" = "1" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --test"
fi

TRAIN_CMD="poetry run view_invariance $TRAIN_ARGS"
echo $TRAIN_CMD

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo Train job failed
    exit 1
fi
