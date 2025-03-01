#!/bin/bash

for seed in $(seq 42 44);
do
    # Normal
    sbatch ./scripts/train_job_view_invariance.sh 1 embed vindr 'mlo cc' 'mlo cc' $seed resnet 0 0 20
    sbatch ./scripts/train_job_view_invariance.sh 1 vindr embed 'mlo cc' 'mlo cc' $seed resnet 0 0 20
    # Invariant
    sbatch ./scripts/train_job_view_invariance.sh 2 embed vindr 'mlo cc' 'mlo cc' $seed resnet 1 0 10
    sbatch ./scripts/train_job_view_invariance.sh 2 vindr embed 'mlo cc' 'mlo cc' $seed resnet 1 0 10
done
