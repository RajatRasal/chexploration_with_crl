#!/bin/bash

for seed in $(seq 42 44);
do
    # Dataset transfer exps -- normal
    sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet 0 0 20

    # Attribute transfer exps -- normal
    sbatch ./scripts/train_job_race_invariance.sh 1 chexpert chexpert '0 1' '2' $seed resnet 0 0 20
    sbatch ./scripts/train_job_race_invariance.sh 1 chexpert chexpert '0 2' '1' $seed resnet 0 0 20
    sbatch ./scripts/train_job_race_invariance.sh 1 chexpert chexpert '2 1' '0' $seed resnet 0 0 20

    # Dataset transfer exps -- invariant
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet 1 0 10

    # Attribute transfer exps -- invariant
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert chexpert '0 1' '2' $seed resnet 1 0 10
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert chexpert '0 2' '1' $seed resnet 1 0 10
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert chexpert '2 1' '0' $seed resnet 1 0 10

    # Dataset transfer exps -- invariant more samples
    sbatch ./scripts/train_job_race_invariance.sh 5 chexpert mimic '0 1 2' '0 1 2' $seed resnet 1 0 10

    # Attribute transfer exps -- invariant more samples
    sbatch ./scripts/train_job_race_invariance.sh 5 chexpert chexpert '0 1' '2' $seed resnet 1 0 10
    sbatch ./scripts/train_job_race_invariance.sh 5 chexpert chexpert '0 2' '1' $seed resnet 1 0 10
    sbatch ./scripts/train_job_race_invariance.sh 5 chexpert chexpert '2 1' '0' $seed resnet 1 0 10
done
