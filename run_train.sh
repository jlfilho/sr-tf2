#!/bin/bash

# ESPCN Train 2x
python train.py --model espcn --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# ESPCN Train 2x, load weights, lr 1e-4
python train.py --model espcn --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10
# ESPCN 4x, transfer learning, lr 10e-3 
python train.py --model espcn --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
sleep 10
# ESPCN 4x, load weights, lr 10e-4
python train.py --model espcn --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10

# G_RTRSGAN Train 2x
python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# G_RTRSGAN Train 2x, load weights, lr 1e-4
python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10
# G_RTRSGAN 4x, transfer learning, lr 10e-3 
python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
sleep 10
# G_RTRSGAN 4x, load weights, lr 10e-4
python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10

# G_ERTSRGAN Train 2x
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# G_ERTSRGAN Train 2x, load weights, lr 1e-4
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10

# G_ERTSRGAN Train 3x
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# G_ERTSRGAN Train 3x, load weights, lr 1e-4
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10


# G_ERTSRGAN 4x, transfer learning, lr 10e-3 
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
sleep 10
# G_ERTSRGAN 4x, load weights, lr 10e-4
python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10

# Parou aqui
# RTSRGAN Train 2x
python train.py --model rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
sleep 10
# RTSRGAN Train 2x, load weights, lr 1e-4
python train.py --model rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
sleep 10
# RTSRGAN 4x, transfer learning, lr 10e-3 
python train.py --model rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
sleep 10
# RTSRGAN 4x, load weights, lr 10e-4
python train.py --model rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
sleep 10

# ERTSRGAN Train 2x
python train.py --model ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# ERTSRGAN Train 2x, load weights, lr 1e-4
python train.py --model ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10
# ERTSRGAN Train 3x
python train.py --model ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
sleep 10
# ERTSRGAN Train 3x, load weights, lr 1e-4
python train.py --model ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
sleep 10

# ERTSRGAN 4x, transfer learning, lr 10e-3 
python train.py --model ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
sleep 10
# ERTSRGAN 4x, load weights, lr 10e-4
python train.py --model ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
