#!/bin/bash
./scripts/train_job.sh 1 chexpert chexpert '0 1' '2' 42 densenet 0 1

if [ $? -ne 0 ]; then
    echo "Test 1 failed"
    exit 1
else
    echo "Test 1 done"
fi
echo

./scripts/train_job.sh 2 chexpert chexpert '0 1' '2' 42 densenet 1 1

if [ $? -ne 0 ]; then
    echo "Test 2 failed"
    exit 1
else
    echo "Test 2 done"
fi
echo

./scripts/train_job.sh 1 chexpert mimic '0 1 2' '0 1 2' 42 densenet 0 1

if [ $? -ne 0 ]; then
    echo "Test 3 failed"
    exit 1
else
    echo "Test 3 done"
fi
echo

./scripts/train_job.sh 1 mimic chexpert '0 1 2' '0 1 2' 42 densenet 1 1

if [ $? -ne 0 ]; then
    echo "Test 4 failed"
    exit 1
else
    echo "Test 4 done"
fi
echo

./scripts/train_job.sh 2 chexpert chexpert '0 2' '1' 42 densenet 1 1

if [ $? -ne 0 ]; then
    echo "Test 5 failed"
    exit 1
else
    echo "Test 5 done"
fi
echo

./scripts/train_job.sh 2 chexpert chexpert '2 1' '0' 42 densenet 1 1

if [ $? -ne 0 ]; then
    echo "Test 6 failed"
    exit 1
else
    echo "Test 6 done"
fi
echo
