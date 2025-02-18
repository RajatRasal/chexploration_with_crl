#!/bin/bash

for seed in $(seq 42 44);
do
    # Normal
    sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet 0 0 20
    sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet 0 0 20
    # Invariance
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet 1 0 5
    sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet 1 0 5
done
