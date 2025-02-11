#!/bin/bash
# ./scripts/train_job.sh 1 chexpert chexpert '0 1' '2' 0 1
# ./scripts/train_job.sh 1 chexpert chexpert '0 2' '1' 0 1
# ./scripts/train_job.sh 1 chexpert chexpert '2 1' '0' 0 1
# 
# ./scripts/train_job.sh 2 chexpert chexpert '0 1' '2' 1 1
./scripts/train_job.sh 2 chexpert chexpert '0 2' '1' 1 1
./scripts/train_job.sh 2 chexpert chexpert '2 1' '0' 1 1