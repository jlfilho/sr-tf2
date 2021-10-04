#!/bin/bash

python train.py --model espcn --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model rtvsrsnt --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model rtvsrgan --load_weights --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model g_rtsrgan --batch_size 16 --learning_rate 1e-4 --num_epochs 1000

sleep 10

python train.py --model rtsrgan --batch_size 16 --load_weights --learning_rate 1e-4 --num_epochs 1000