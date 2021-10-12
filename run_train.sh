#!/bin/bash

python train.py --model espcn --batch_size 64 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5

sleep 10

python train.py --model g_rtsrgan --batch_size 32 --learning_rate 1e-2 --lr_decay_epochs 10 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5

python train.py --model g_rtsrgan --batch_size 64 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5


sleep 10

python train.py --model g_ertsrgan --batch_size 32 --learning_rate 1e-2 --lr_decay_epochs 10 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5

sleep 10

python train.py --model rtsrgan --load_weights --batch_size 32 --learning_rate 1e-2 --lr_decay_epochs 10 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5

sleep 10

python train.py --model ertsrgan --load_weights --batch_size 32 --learning_rate 1e-2 --lr_decay_epochs 10 --num_epochs 100 --steps_per_epochs 100 --epochs_per_save 5
