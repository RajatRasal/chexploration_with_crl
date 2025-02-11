#!/bin/bash

####### Densenet
# Dataset transfer exps -- normal
sbatch ./scripts/train_job.sh 1 chexpert mimic '0 1 2' '0 1 2' 42 densenet 0 0
sbatch ./scripts/train_job.sh 1 mimic chexpert '0 1 2' '0 1 2' 42 densenet 0 0

# Dataset transfer exps -- invariant
sbatch ./scripts/train_job.sh 2 chexpert mimic '0 1 2' '0 1 2' 42 densenet 1 0
sbatch ./scripts/train_job.sh 2 mimic chexpert '0 1 2' '0 1 2' 42 densenet 1 0

# Attribute transfer exps -- normal
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 1' '2' 42 densenet 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 2' '1' 42 densenet 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '2 1' '0' 42 densenet 0 0

# Attribute transfer exps -- invariant
sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 1' '2' 42 densenet 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 2' '1' 42 densenet 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '2 1' '0' 42 densenet 1 0

####### ViTB16
# Dataset transfer exps -- normal
sbatch ./scripts/train_job.sh 1 chexpert mimic '0 1 2' '0 1 2' 42 vitb16 0 0
sbatch ./scripts/train_job.sh 1 mimic chexpert '0 1 2' '0 1 2' 42 vitb16 0 0

# Dataset transfer exps -- invariant
sbatch ./scripts/train_job.sh 2 chexpert mimic '0 1 2' '0 1 2' 42 vitb16 1 0
sbatch ./scripts/train_job.sh 2 mimic chexpert '0 1 2' '0 1 2' 42 vitb16 1 0

# Attribute transfer exps -- normal
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 1' '2' 42 vitb16 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 2' '1' 42 vitb16 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '2 1' '0' 42 vitb16 0 0

# Attribute transfer exps -- invariant
sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 1' '2' 42 vitb16 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 2' '1' 42 vitb16 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '2 1' '0' 42 vitb16 1 0

####### ResNet
# MCC exps
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 1' '2' 42 resnet 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '0 2' '1' 43 resnet 0 0
sbatch ./scripts/train_job.sh 1 chexpert chexpert '2 1' '0' 44 resnet 0 0

sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 1' '2' 42 resnet 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '0 2' '1' 43 resnet 1 0
sbatch ./scripts/train_job.sh 2 chexpert chexpert '2 1' '0' 44 resnet 1 0
