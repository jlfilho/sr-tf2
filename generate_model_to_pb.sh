#!/bin/bash

python generate_model_to_pb.py --model ertsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x2/ --ckpt_path ./checkpoint/g_ertsrgan_2x/ --scale_factor 2

python generate_model_to_pb.py --model ertsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x3/ --ckpt_path ./checkpoint/g_ertsrgan_3x/ --scale_factor 3

python generate_model_to_pb.py --model ertsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/g_ertsrgan_4x/ --scale_factor 4



python generate_model_to_pb.py --model escpn --output_folder dnn_bin_models/ --ckpt_path ./checkpoint/3X/270p_qp17/espcn_3x/ --scale_factor 3
