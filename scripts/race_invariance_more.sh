#!/bin/bash

for seed in $(seq 44 49);
do
    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet32 0 0 20
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet32 0 0 20
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet32 1 0 10
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet32 1 0 10

    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 0 0 20
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 0 0 20
    # ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 0 0 5 1 512
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 0 0 5 -1 512
    # Invariance
    # ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 5 0.1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 5 1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 5 0.1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 5 0.01 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 1 0 5

    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed densenet 0 0 20
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed densenet 0 0 20
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed densenet 1 0 10
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed densenet 1 0 10

    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet50 0 0 20
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet50 0 0 20
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet50 1 0 10
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet50 1 0 10


    # # VIEW INVARIANCE
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed resnet18 0 0 5 '-1' 512
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed resnet18 0 0 5 '-1' 512
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 5 1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 5 0.1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 5 1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 5 0.1 512

    # # VIEW INVARIANCE
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed resnet18 0 0 20 '-1' 512
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed resnet18 0 0 20 '-1' 512
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 10 1 512
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 10 1 512

    # # VIEW INVARIANCE
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed densenet 0 0 20 '-1' 64
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed densenet 0 0 20 '-1' 64
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 1 0 10 1 64
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 1 0 10 1 64

    # VIEW INVARIANCE
    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32

    # # RACE INVARIANCE
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed efficientnetb0 0 0 20 '-1' 32 Cardiomegaly race
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed efficientnetb0 0 0 20 '-1' 32 Cardiomegaly race
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly race
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 0 0 20 '-1' 32 Cardiomegaly race
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 0 0 20 '-1' 32 Cardiomegaly race
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 10 1 32 Cardiomegaly race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 1 0 10 1 32 Cardiomegaly race

    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed efficientnetb0 0 0 20 '-1' 32 'Pleural Effusion' race
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed efficientnetb0 0 0 20 '-1' 32 'Pleural Effusion' race
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' race
    # # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 0 0 20 '-1' 32 'Pleural Effusion' race
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 0 0 20 '-1' 32 'Pleural Effusion' race
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed resnet18 1 0 10 1 32 'Pleural Effusion' race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed resnet18 1 0 10 1 32 'Pleural Effusion' race

    # SEX INVARIANCE
    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32 Cardiomegaly sex
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32 Cardiomegaly sex
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly sex
    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed resnet18 0 0 20 '-1' 32 Cardiomegaly sex
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed resnet18 0 0 20 '-1' 32 Cardiomegaly sex
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 10 1 32 Cardiomegaly sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 10 1 32 Cardiomegaly sex

    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32 'Pleural Effusion' sex
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed efficientnetb0 0 0 20 '-1' 32 'Pleural Effusion' sex
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' sex
    # Normal
    # sbatch ./scripts/train_job_race_invariance.sh 1 chexpert mimic '0 1' '0 1' $seed resnet18 0 0 20 '-1' 32 'Pleural Effusion' sex
    # sbatch ./scripts/train_job_race_invariance.sh 1 mimic chexpert '0 1' '0 1' $seed resnet18 0 0 20 '-1' 32 'Pleural Effusion' sex
    # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 10 1 32 'Pleural Effusion' sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 10 1 32 'Pleural Effusion' sex

    # # VIEW INVARIANCE
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 Cardiomegaly view
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 10 1 32 Cardiomegaly view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 10 1 32 Cardiomegaly view

    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed efficientnetb0 1 0 10 1 32 'Pleural Effusion' view
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed resnet18 1 0 10 1 32 'Pleural Effusion' view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed resnet18 1 0 10 1 32 'Pleural Effusion' view

    # NO INVARIANCE
    # Invariance
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 0 0 10 1 32 Cardiomegaly sex
    sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 0 0 10 1 32 Cardiomegaly sex
    # Invariance
    sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 0 0 10 1 32 'Pleural Effusion' sex
    sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 0 0 10 1 32 'Pleural Effusion' sex

    # # VIEW INVARIANCE
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 1 0 10 1 32 Cardiomegaly view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 1 0 10 1 32 Cardiomegaly view
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 1 0 10 1 32 'Pleural Effusion' view
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 1 0 10 1 32 'Pleural Effusion' view

    # # SEX INVARIANCE
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 1 0 10 1 32 Cardiomegaly sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 1 0 10 1 32 Cardiomegaly sex
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1' '0 1' $seed densenet 1 0 10 1 32 'Pleural Effusion' sex
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1' '0 1' $seed densenet 1 0 10 1 32 'Pleural Effusion' sex

    # # RACE INVARIANCE
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed densenet 1 0 10 1 32 Cardiomegaly race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed densenet 1 0 10 1 32 Cardiomegaly race
    # # Invariance
    # sbatch ./scripts/train_job_race_invariance.sh 2 chexpert mimic '0 1 2' '0 1 2' $seed densenet 1 0 10 1 32 'Pleural Effusion' race
    # sbatch ./scripts/train_job_race_invariance.sh 2 mimic chexpert '0 1 2' '0 1 2' $seed densenet 1 0 10 1 32 'Pleural Effusion' race
done
