#!/bin/bash

# # ESPCN Train 2x
# python train.py --model espcn --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # ESPCN Train 2x, load weights, lr 1e-4
# python train.py --model espcn --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10
# # ESPCN 4x, transfer learning, lr 10e-3 
# python train.py --model espcn --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
# sleep 10
# # ESPCN 4x, load weights, lr 10e-4
# python train.py --model espcn --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10

# # G_RTRSGAN Train 2x
# python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # G_RTRSGAN Train 2x, load weights, lr 1e-4
# python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10
# # G_RTRSGAN 4x, transfer learning, lr 10e-3 
# python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
# sleep 10
# # G_RTRSGAN 4x, load weights, lr 10e-4
# python train.py --model g_rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10

# # G_ERTSRGAN Train 2x
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # G_ERTSRGAN Train 2x, load weights, lr 1e-4
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10

# # G_ERTSRGAN Train 3x
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # G_ERTSRGAN Train 3x, load weights, lr 1e-4
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10


# # G_ERTSRGAN 4x, transfer learning, lr 10e-3 
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
# sleep 10
# # G_ERTSRGAN 4x, load weights, lr 10e-4
# python train.py --model g_ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10

# # Parou aqui
# # RTSRGAN Train 2x
# python train.py --model rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
# sleep 10
# # RTSRGAN Train 2x, load weights, lr 1e-4
# python train.py --model rtsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
# sleep 10
# # RTSRGAN 4x, transfer learning, lr 10e-3 
# python train.py --model rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
# sleep 10
# # RTSRGAN 4x, load weights, lr 10e-4
# python train.py --model rtsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights --trainable_layer final
# sleep 10

# # ERTSRGAN Train 2x
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # ERTSRGAN Train 2x, load weights, lr 1e-4
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/2x/dataset_info.txt --train_dataset_path datasets/train/2x/dataset.tfrecords --valid_dataset_info_path datasets/test/2x/dataset_info.txt --valid_dataset_path datasets/test/2x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10
# # ERTSRGAN Train 3x
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5
# sleep 10
# # ERTSRGAN Train 3x, load weights, lr 1e-4
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/3x/dataset_info.txt --train_dataset_path datasets/train/3x/dataset.tfrecords --valid_dataset_info_path datasets/test/3x/dataset_info.txt --valid_dataset_path datasets/test/3x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights
# sleep 10

# # ERTSRGAN 4x, transfer learning, lr 10e-3 
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-3 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --transfer_learning
# sleep 10
# # ERTSRGAN 4x, load weights, lr 10e-4
# python train.py --model ertsrgan --train_dataset_info_path datasets/train/4x/dataset_info.txt --train_dataset_path datasets/train/4x/dataset.tfrecords --valid_dataset_info_path datasets/test/4x/dataset_info.txt --valid_dataset_path datasets/test/4x/dataset.tfrecords --batch_size 32 --learning_rate 1e-4 --lr_decay_epochs 20 --num_epochs 10 --steps_per_epochs 100 --epochs_per_save 5 --load_weights




python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/810p/ \
--test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ \
--ckpt_path checkpoint/tmp/ \
--path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30



# ESPCN Train 3x
python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/810p/ \
--test_logdir  test_logdir/3X/270p_qp28/ \
--logdir logdir/3X/270p_qp28/ \
--ckpt_path checkpoint/3X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/810p/ \
--test_logdir  test_logdir/3X/270p_qp28/ \
--logdir logdir/3X/270p_qp28/ \
--ckpt_path checkpoint/3X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_ertsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/3X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/3X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/3X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/810p/ \
--test_logdir  test_logdir/3X/270p_qp28/ \
--logdir logdir/3X/270p_qp28/ \
--ckpt_path checkpoint/3X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

# -------------------------------
# ESPCN Train 4x
python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp17/ \
--logdir logdir/4X/270p_qp17/ \
--ckpt_path checkpoint/4X/270p_qp17/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp17/ \
--logdir logdir/4X/270p_qp17/ \
--ckpt_path checkpoint/4X/270p_qp17/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_ertsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp17/ \
--logdir logdir/4X/270p_qp17/ \
--ckpt_path checkpoint/4X/270p_qp17/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

# -------------------------------
# ESPCN Train 4x
python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp20/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp20/ \
--logdir logdir/4X/270p_qp20/ \
--ckpt_path checkpoint/4X/270p_qp20/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp20/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp20/ \
--logdir logdir/4X/270p_qp20/ \
--ckpt_path checkpoint/4X/270p_qp20/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_ertsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp20/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp20/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp20/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp20/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp20/ \
--logdir logdir/4X/270p_qp20/ \
--ckpt_path checkpoint/4X/270p_qp20/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

# -------------------------------
# ESPCN Train 4x
python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp28/ \
--logdir logdir/4X/270p_qp28/ \
--ckpt_path checkpoint/4X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp28/ \
--logdir logdir/4X/270p_qp28/ \
--ckpt_path checkpoint/4X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10

python3 train.py --model g_ertsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/train/4X/270p_qp28/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/val/4X/270p_qp28/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/test/4X/270p_qp28/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/lr/270p_qp28/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/hr/1080p/ \
--test_logdir  test_logdir/4X/270p_qp28/ \
--logdir logdir/4X/270p_qp28/ \
--ckpt_path checkpoint/4X/270p_qp28/ \
--hot_test_size 8 \
--batch_size 32 \
--learning_rate 1e-3 \
--lr_decay_epochs 20 \
--type_reduce_lr schedules \
--num_epochs 200 \
--steps_per_epoch 100 \
--epochs_per_save 30

sleep 10