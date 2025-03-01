#!/bin/bash

##### Resnet (mlo, cc)
# Attribute transfer exps -- normal
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'mlo cc' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'mlo cc' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo cc' 'mlo cc' 42 resnet 0 0

sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo cc' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo cc' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo cc' 42 resnet 1 0
# 
# ##### Resnet (mlo -> cc)
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo' 'cc' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo' 'cc' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'mlo' 'cc' 42 resnet 0 0
# 
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'cc' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'cc' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'cc' 42 resnet 1 0
# 
# ##### Resnet (cc -> mlo)
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'cc' 'mlo' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'cc' 'mlo' 42 resnet 0 0
# sbatch ./scripts/train_job_view_invariance.sh 1 embed embed 'cc' 'mlo' 42 resnet 0 0
# 
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo' 42 resnet 1 0
# sbatch ./scripts/train_job_view_invariance.sh 2 embed embed 'mlo cc' 'mlo' 42 resnet 1 0
