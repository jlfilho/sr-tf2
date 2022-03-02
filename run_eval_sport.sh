#!/bin/bash

# 2) Second evaluate quantitative and perceptual-oriented with content-aware on sport dataset;  

# python3 train.py --model teacher \
# --train_dataset_info_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/sport/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
# --num_epochs 100 --steps_per_epoch 200 --epochs_per_save 50 --loss_fn mae --load_weights

mkdir test_logdir/test/sport/percep
mkdir test_logdir/test/sport/quant
mkdir test_logdir/test/sport/hr
mkdir test_logdir/test/sport/bicubic
mkdir test_logdir/test/sport/espcn


python3 train.py --model espcn \
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
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mse --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/espcn test_logdir/test/sport/quant/
mkdir test_logdir/test/sport/espcn

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
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/espcn test_logdir/test/sport/percep/
mkdir test_logdir/test/sport/imdn

python3 train.py --model imdn \
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
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mse --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/imdn test_logdir/test/sport/quant/
mkdir test_logdir/test/sport/imdn

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
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/imdn test_logdir/test/sport/percep/
mkdir test_logdir/test/sport/g_rtsrgan

python3 train.py --model g_rtsrgan \
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
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mse --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/g_rtsrgan test_logdir/test/sport/quant/
mkdir test_logdir/test/sport/g_rtsrgan

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
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/g_rtsrgan test_logdir/test/sport/percep/
mkdir test_logdir/test/sport/g_rtvsrgan

The best g_rtvsrgan
python3 train.py --model g_rtvsrgan \
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
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mse --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/g_rtvsrgan test_logdir/test/sport/quant/
mkdir test_logdir/test/sport/g_rtvsrgan

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
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 50 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/g_rtvsrgan test_logdir/test/sport/percep/
mkdir test_logdir/test/sport/evsrnet

python3 train.py --model evsrnet \
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
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/evsrnet test_logdir/test/sport/quant/
mkdir test_logdir/test/sport/evsrnet

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
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules \
--num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 1e-3 5e-1 1e-3 1e-1 --eval --test_cluster sport --range_to_save 10

sleep 10
mv test_logdir/test/sport/evsrnet test_logdir/test/sport/percep/
cp -R test_logdir/test/sport/bicubic test_logdir/test/sport/percep/
cp -R test_logdir/test/sport/hr test_logdir/test/sport/percep/
mv test_logdir/test/sport/bicubic test_logdir/test/sport/quant/
mv test_logdir/test/sport/hr test_logdir/test/sport/quant/
