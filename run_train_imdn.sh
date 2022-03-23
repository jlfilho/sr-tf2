#!/bin/bash


# # 2x
# echo "--model teacher generic"
# python3 train.py --model teacher \
# --train_dataset_info_path datasets/loaded_harmonic/output/generic/train/2X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/generic/train/2X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/generic/val/2X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/generic/val/2X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/generic/test/2X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/generic/test/2X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/540p/ --test_logdir  test_logdir/2X/270p_qp17/generic/ \
# --logdir logdir/2X/270p_qp17/ --ckpt_path checkpoint/2X/ --path_to_eval test_logdir/2X/270p_qp17/generic/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae

# sleep 10

# cp -R checkpoint/2X/teacher_2x checkpoint/2X/generic/ 

# echo "--model teacher game" 

# python3 train.py --model teacher \
# --train_dataset_info_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset_info.txt \
# --train_dataset_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset.tfrecords \
# --val_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
# --val_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
# --test_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
# --test_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
# --lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
# --hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/540p/ --test_logdir  test_logdir/2X/270p_qp17/game/ \
# --logdir logdir/2X/270p_qp17/ --ckpt_path checkpoint/2X/ --path_to_eval test_logdir/2X/270p_qp17/game/stats.txt \
# --hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
# --num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae --load_weights

# sleep 10

# cp -R checkpoint/2X/teacher_2x checkpoint/2X/game/ 

echo "--model imdn generic"

python3 train.py --model imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/generic/train/2X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/generic/train/2X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/generic/val/2X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/generic/val/2X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/generic/test/2X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/generic/test/2X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/generic/hr/540p/ --test_logdir  test_logdir/2X/270p_qp17/generic/ \
--logdir logdir/2X/270p_qp17/ --ckpt_path checkpoint/2X/ --path_to_eval test_logdir/2X/270p_qp17/generic/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse

sleep 10

cp -R checkpoint/2X/imdn_2x checkpoint/2X/generic/ 

echo "--model imdn game"

python3 train.py --model imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/540p/ --test_logdir  test_logdir/2X/270p_qp17/game/ \
--logdir logdir/2X/270p_qp17/ --ckpt_path checkpoint/2X/ --path_to_eval test_logdir/2X/270p_qp17/game/stats.txt \
--hot_test_size 8 --batch_size 32 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mse --load_weights

sleep 10

cp -R checkpoint/2X/imdn_2x checkpoint/2X/game/ 

echo "--model percsr imdn game"

python3 train.py --model percsr \
--generator imdn \
--train_dataset_info_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset_info.txt \
--train_dataset_path datasets/loaded_harmonic/output/game/train/2X/270p_qp17/dataset.tfrecords \
--val_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
--val_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
--test_dataset_info_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset_info.txt \
--test_dataset_path datasets/loaded_harmonic/output/game/test/2X/270p_qp17/dataset.tfrecords \
--lr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/ \
--hr_hot_test_path datasets/loaded_harmonic/img_hot_test/game/hr/540p/ --test_logdir  test_logdir/2X/270p_qp17/game/ \
--logdir logdir/2X/270p_qp17/ --ckpt_path checkpoint/2X/ --path_to_eval test_logdir/2X/270p_qp17/game/stats.txt \
--hot_test_size 8 --batch_size 8 --learning_rate 1e-3 --type_reduce_lr schedules --schedule_values 100 200 \
--num_epochs 200 --steps_per_epoch 100 --epochs_per_save 100 --loss_fn mae \
--list_weights 2e-3 5e-5 1e-5 7e-3 --load_weights 

sleep 10

cp -R checkpoint/2X/percsr_2x checkpoint/2X/game/ 

python generate_model_to_pb.py --model imdn --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/2X/ --ckpt_path ./checkpoint/2X/game/percsr_2x/imdn/generator/ --scale_factor 2

