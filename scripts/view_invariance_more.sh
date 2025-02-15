#!/bin/bash

for seed in $(seq 42 44);
do
    # (mlo, cc) -> cc
    # (mlo, cc) -> mlo
    # Attribute transfer exps -- normal
    sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'cc' $seed resnet 0 0 20
    sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'mlo' $seed resnet 0 0 20
    # Attribute transfer exps -- invariant
    sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'cc' $seed resnet 1 0 5
    sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo' $seed resnet 1 0 5

    # (mlo, cc) -> (mlo, cc)
    # Attribute transfer exps -- normal
    sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'mlo cc' $seed resnet 0 0 20
    # Attribute transfer exps -- invariant
    sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo cc' $seed resnet 1 0 5
done
