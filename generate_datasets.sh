#!/bin/bash

# python3 datasets/load_harmonic_video.py --save_folder=datasets/loaded_harmonic
# python3 datasets/load_div2k.py --save_folder=datasets/loaded_div2k

# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic --dataset_folder=datasets/train --type=blocks --temporal_radius=1 --frames_per_scene=2 --block_size=36 --stride=36 --crop --scene_changes=datasets/scene_changes_harmonic.json --block_min_std=20.0
# python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/train --dataset_folder=datasets/train_div2k --type=blocks --temporal_radius=1 --block_size=36 --stride=36
# python3 datasets/prepare_div2k_dataset.py --div2k_folder=datasets/loaded_div2k/test --dataset_folder=datasets/test_div2k --type=full --temporal_radius=1

# mkdir datasets/train_merged
# cat datasets/train_div2k/dataset.tfrecords datasets/train/dataset.tfrecords > datasets/train_merged/dataset.tfrecords
# echo $(($(sed -n 1p datasets/train/dataset_info.txt) + $(sed -n 1p datasets/train_div2k/dataset_info.txt))) > datasets/train_merged/dataset_info.txt
# tail -n +2 datasets/train/dataset_info.txt >> datasets/train_merged/dataset_info.txt



# # generate dataset to train and test with video
# #2x
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/train/ --dataset_folder=datasets/train/2x --scale_factor=2 --type=blocks --temporal_radius=1 --frames_per_scene=4 --block_size=36 --stride=36 --crop --scene_changes=datasets/scene_changes_harmonic.json --block_min_std=20.0
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/test/ --dataset_folder=datasets/test/2x --scale_factor=2 --type=full --temporal_radius=1 --frames_per_scene=4 --scene_changes=datasets/scene_changes_harmonic_test.json 



# #3x
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/train/ --dataset_folder=datasets/train/3x --scale_factor=3 --type=blocks --temporal_radius=1 --frames_per_scene=4 --block_size=36 --stride=36 --crop --scene_changes=datasets/scene_changes_harmonic.json --block_min_std=20.0
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/test/ --dataset_folder=datasets/test/3x --scale_factor=3 --type=full --temporal_radius=1 --frames_per_scene=4 --scene_changes=datasets/scene_changes_harmonic_test.json 


# #4x
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/train/ --dataset_folder=datasets/train --scale_factor=4 --type=blocks --temporal_radius=1 --frames_per_scene=4 --block_size=36 --stride=36 --crop --scene_changes=datasets/scene_changes_harmonic.json --block_min_std=20.0
# python3 datasets/prepare_dataset.py --video_folder=datasets/loaded_harmonic/test/ --dataset_folder=datasets/test --scale_factor=4 --type=full --temporal_radius=1 --frames_per_scene=4 --scene_changes=datasets/scene_changes_harmonic_test.json 



# With images
# train 2X
# echo "train 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/540p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/540p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/540p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/720p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/360p_qp17 --type=blocks --lr_prefix=360p_qp17 --hr_prefix=720p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/720p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/360p_qp20 --type=blocks --lr_prefix=360p_qp20 --hr_prefix=720p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/720p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/360p_qp28 --type=blocks --lr_prefix=360p_qp28 --hr_prefix=720p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/540p_qp17 --type=blocks --lr_prefix=540p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/540p_qp20 --type=blocks --lr_prefix=540p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/2X/540p_qp28 --type=blocks --lr_prefix=540p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2 


# # test 2X
# echo "test 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/540p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/540p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/540p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/720p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/720p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/720p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/2X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 


# # val 2X
# echo "val 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/540p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/540p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/540p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/720p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/720p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/720p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=720p --temporal_radius=1 --crop_height=720 --crop_width=1280 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/2X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=2 


# # train 3X
# echo "train 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/810p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/810p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/810p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/360p_qp17 --type=blocks --lr_prefix=360p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/360p_qp20 --type=blocks --lr_prefix=360p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/360p_qp28 --type=blocks --lr_prefix=360p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1620p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/540p_qp17 --type=blocks --lr_prefix=540p_qp17 --hr_prefix=1620p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1620p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/540p_qp20 --type=blocks --lr_prefix=540p_qp20 --hr_prefix=1620p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1620p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/3X/540p_qp28 --type=blocks --lr_prefix=540p_qp28 --hr_prefix=1620p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3 


# # test 3X
# echo "test 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/810p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/810p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/810p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1620p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1620p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1620p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/3X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 


# # val 3X
# echo "val 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/810p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/810p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/810p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1620p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1620p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1620p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/3X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=1620p --temporal_radius=1 --crop_height=1620 --crop_width=2880 --scale_factor=3 

# # train 4X
# echo "train 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1080p --lr_folder=datasets/loaded_harmonic/img_train/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1440p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/360p_qp17 --type=blocks --lr_prefix=360p_qp17 --hr_prefix=1440p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1440p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/360p_qp20 --type=blocks --lr_prefix=360p_qp20 --hr_prefix=1440p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/1440p --lr_folder=datasets/loaded_harmonic/img_train/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/360p_qp28 --type=blocks --lr_prefix=360p_qp28 --hr_prefix=1440p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/2160p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/540p_qp17 --type=blocks --lr_prefix=540p_qp17 --hr_prefix=2160p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/2160p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/540p_qp20 --type=blocks --lr_prefix=540p_qp20 --hr_prefix=2160p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_train/hr/2160p --lr_folder=datasets/loaded_harmonic/img_train/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/train/4X/540p_qp28 --type=blocks --lr_prefix=540p_qp28 --hr_prefix=2160p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4 


# # test 4X
# echo "test 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1080p --lr_folder=datasets/loaded_harmonic/img_test/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1440p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1440p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/1440p --lr_folder=datasets/loaded_harmonic/img_test/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/2160p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/2160p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_test/hr/2160p --lr_folder=datasets/loaded_harmonic/img_test/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 


## python3 datasets/prepare_div2k_dataset.py --hr_folder=/home/joao/data/img_test/hr/2160p --lr_folder=/home/joao/data/img_test/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

## python3 datasets/prepare_div2k_dataset.py --hr_folder=/home/joao/data/img_test/hr/2160p --lr_folder=/home/joao/data/img_test/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/test/4X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 


# val 4X
# echo "val 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1080p --lr_folder=datasets/loaded_harmonic/img_val/lr/270p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1440p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/360p_qp17 --type=full --lr_prefix=360p_qp17 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1440p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/360p_qp20 --type=full --lr_prefix=360p_qp20 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/1440p --lr_folder=datasets/loaded_harmonic/img_val/lr/360p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/360p_qp28 --type=full --lr_prefix=360p_qp28 --hr_prefix=1440p --temporal_radius=1 --crop_height=1440 --crop_width=2560 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/2160p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp17/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/540p_qp17 --type=full --lr_prefix=540p_qp17 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/2160p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp20/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/540p_qp20 --type=full --lr_prefix=540p_qp20 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

# python3 datasets/prepare_div2k_dataset.py --hr_folder=datasets/loaded_harmonic/img_val/hr/2160p --lr_folder=datasets/loaded_harmonic/img_val/lr/540p_qp28/ --dataset_folder=datasets/loaded_harmonic/output/val/4X/540p_qp28 --type=full --lr_prefix=540p_qp28 --hr_prefix=2160p --temporal_radius=1 --crop_height=2160 --crop_width=3840 --scale_factor=4 

#########################

# # With images
# # train 2X
# # echo "train 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/2X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/2X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/2X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2


# # With images
# # train 3X
# # echo "train 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/3X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/3X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/3X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3


# # With images
# # train 4X
# # echo "train 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4



#online
# With images
# train 4X
# echo "tain 4X"

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp17 \
--dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp17_ol --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp20 \
--dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp20_ol --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp28 \
--dataset_folder=datasets/loaded_harmonic/output/game/train/4X/270p_qp28_ol --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4


#online
# With images
# train 4X
# echo "tain 4X"

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp17 \
--dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp17_ol --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp20 \
--dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp20_ol --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp28 \
--dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp28_ol --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4


#online
# With images
# train 4X
# echo "tain 4X"

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp17 \
--dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp17_ol --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp20 \
--dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp20_ol --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp28 \
--dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp28_ol --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4


# #-----------------------
# # With images
# # train 2X
# # echo "train 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/2X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/2X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/2X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2


# # With images
# # train 3X
# # echo "train 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/3X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/3X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/3X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3


# # With images
# # train 4X
# # echo "train 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/train/4X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4


# #-----------------------
# # With images
# # train 2X
# # echo "train 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/2X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/2X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/2X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=2



# # With images
# # train 3X
# # echo "train 3X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/3X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/3X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/3X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=3



# # With images
# # train 4X
# # echo "train 4X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp17 --type=blocks --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp20 --type=blocks --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_train/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/train/4X/270p_qp28 --type=blocks --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --block_size=36 --stride=36 --scale_factor=4



# # With images
# # test 2X
# # echo "test 2X"
# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/2X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/2X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/2X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2


# # With images
# # test 3X
# # echo "test 3X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/3X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/3X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/3X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3


# # With images
# # test 4X
# # echo "test 4X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/4X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/4X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/game/test/4X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4


# # With images
# # test 2X
# # echo "test 2X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/2X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/2X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/2X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2


# # With images
# # test 3X
# # echo "test 3X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/3X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/3X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/3X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3


# # With images
# # test 4X
# # echo "test 4X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/4X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/4X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/podcast/test/4X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# # With images
# # test 2X
# # echo "test 2X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/2X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/2X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/540p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/2X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=540p --temporal_radius=1 --crop_height=540 --crop_width=960 --scale_factor=2


# # With images
# # test 3X
# # echo "test 3X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/3X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/3X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/810p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/3X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=810p --temporal_radius=1 --crop_height=810 --crop_width=1440 --scale_factor=3


# # With images
# # test 4X
# # echo "test 4X"

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp17 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/4X/270p_qp17 --type=full --lr_prefix=270p_qp17 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp20 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/4X/270p_qp20 --type=full --lr_prefix=270p_qp20 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4

# python3 datasets/prepare_div2k_dataset.py --hr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p --lr_folder=/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp28 \
# --dataset_folder=datasets/loaded_harmonic/output/sport/test/4X/270p_qp28 --type=full --lr_prefix=270p_qp28 --hr_prefix=1080p --temporal_radius=1 --crop_height=1080 --crop_width=1920 --scale_factor=4
