#!/bin/bash

# 1) First train quantitative oriented with content-aware on game dataset;  

# 2) Second train perceptual oriented with content-aware on game dataset;

# python3 train.py --model teacher \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 100 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae --load_weights

# python3 train.py --model espcn \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

# sleep 10

# python3 train.py --model percsr \
# --generator espcn \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

# sleep 10

# python3 train.py --model imdn \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

# sleep 10

# python3 train.py --model percsr \
# --generator imdn \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

# sleep 10

# python3 train.py --model g_rtsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

# sleep 10

# python3 train.py --model percsr \
# --generator g_rtsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

# sleep 10

# #The best g_rtvsrgan
# python3 train.py --model g_rtvsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

# sleep 10

# python3 train.py --model percsr \
# --generator g_rtvsrgan \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

# sleep 10

# python3 train.py --model evsrnet \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 4 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 50 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

# sleep 10

# python3 train.py --model percsr \
# --generator evsrnet \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/4X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/1080p/ --test_logdir  test_logdir/tmp/ \
# --logdir logdir/tmp/ --ckpt_path checkpoint/tmp/ --path_to_eval test_logdir/tmp/stats.txt \
# --hot_test_size 4 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
# --list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 



# 3x
python3 train.py --model teacher \
--train_dataset_info_path datasets/loaded_harmonic/output/generic/train/3X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/generic/train/3X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/generic/val/3X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/generic/val/3X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/generic/test/3X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/generic/test/3X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/810p/ --test_logdir  test_logdir/3X/270p_qp17/generic/ \
--logdir logdir/3X/270p_qp17/ --ckpt_path checkpoint/3X/ --path_to_eval test_logdir/3X/270p_qp17/generic/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae

sleep 10

cp -R checkpoint/3X/teacher_3x checkpoint/3X/generic/ 

python3 train.py --model teacher \
--train_dataset_info_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/810p/ --test_logdir  test_logdir/3X/270p_qp17/game/ \
--logdir logdir/3X/270p_qp17/ --ckpt_path checkpoint/3X/ --path_to_eval test_logdir/3X/270p_qp17/game/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae --load_weights

sleep 10

cp -R checkpoint/3X/teacher_3x checkpoint/3X/game/ 


python3 train.py --model imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/generic/train/3X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/generic/train/3X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/generic/val/3X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/generic/val/3X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/generic/test/3X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/generic/test/3X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/810p/ --test_logdir  test_logdir/3X/270p_qp17/generic/ \
--logdir logdir/3X/270p_qp17/ --ckpt_path checkpoint/3X/ --path_to_eval test_logdir/3X/270p_qp17/generic/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse

sleep 10

cp -R checkpoint/3X/imdn_3x checkpoint/3X/generic/ 

python3 train.py --model imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/810p/ --test_logdir  test_logdir/3X/270p_qp17/game/ \
--logdir logdir/3X/270p_qp17/ --ckpt_path checkpoint/3X/ --path_to_eval test_logdir/3X/270p_qp17/game/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

cp -R checkpoint/3X/imdn_3x checkpoint/3X/game/ 

python3 train.py --model percsr \
--generator imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/game/train/3X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/game/test/3X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/810p/ --test_logdir  test_logdir/3X/270p_qp17/game/ \
--logdir logdir/3X/270p_qp17/ --ckpt_path checkpoint/3X/ --path_to_eval test_logdir/3X/270p_qp17/game/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

cp -R checkpoint/3X/percsr_3x checkpoint/3X/game/ 

