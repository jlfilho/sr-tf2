#!/bin/bash

# python3 train.py --model teacher \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 200 --epochs_per_save 100 --loss_fn mae

# python3 train.py --model espcn \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse 

# sleep 10

# python3 train.py --model percsr \
# --generator espcn \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 1e-2 1e-1 1e-2 1e-1 --load_weight 

# sleep 10

# python3 train.py --model imdn \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse 

# sleep 10

python3 train.py --model percsr \
--generator imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --load_weight 

sleep 10

# python3 train.py --model g_rtsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse 

# sleep 10

# python3 train.py --model percsr \
# --generator g_rtsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 1e-2 1e-1 1e-2 1e-1 --load_weight 

# sleep 10

# #The best g_ertsrgan
# python3 train.py --model g_ertsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse 

# sleep 10

python3 train.py --model percsr \
--generator g_ertsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --load_weight 

# sleep 10

# python3 train.py --model evsrnet \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse 

# sleep 10

# python3 train.py --model percsr \
# --generator evsrnet \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 1e-2 1e-1 1e-2 1e-1 --load_weight 

# sleep 10
