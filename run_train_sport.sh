#!/bin/bash

# 1) First train quantitative oriented with content-aware on sport dataset;  

# 2) Second train perceptual oriented with content-aware on sport dataset;

python3 train.py --model teacher \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 200 --epochs_per_save 100 --loss_fn mae --load_weights

python3 train.py --model espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

python3 train.py --model percsr \
--generator espcn \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

python3 train.py --model imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

python3 train.py --model percsr \
--generator imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

python3 train.py --model g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

python3 train.py --model percsr \
--generator g_rtsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

#The best g_rtvsrgan
python3 train.py --model g_rtvsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

python3 train.py --model percsr \
--generator g_rtvsrgan \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

python3 train.py --model evsrnet \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 4 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

python3 train.py --model percsr \
--generator evsrnet \
--train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
--logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
--hot_test_size 4 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 
