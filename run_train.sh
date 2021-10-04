#!/bin/bash


rm -r logdir/*

rm -r checkpoint/*

rm test_logdir/*

python train.py --model espcn --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model rtvsrsnt --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model rtvsrgan --load_weights --learning_rate 1e-4 --num_epochs 1000