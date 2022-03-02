#!/bin/bash

python generate_model_to_pb.py --model rtvsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x2/ --ckpt_path ./checkpoint/g_rtvsrgan_2x/ --scale_factor 2

python generate_model_to_pb.py --model rtvsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x3/ --ckpt_path ./checkpoint/g_rtvsrgan_3x/ --scale_factor 3




python generate_model_to_pb.py --model escpn --output_folder dnn_bin_models/ --ckpt_path ./checkpoint/3X/270p_qp17/espcn_3x/ --scale_factor 3


# X4
python generate_model_to_pb.py --model espcn --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/4X/270p_qp28/espcn_4x/ --scale_factor 4

python generate_model_to_pb.py --model imdn --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path checkpoint/4X/270p_qp28/imdn_4x/ --scale_factor 4

python generate_model_to_pb.py --model g_rtsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path checkpoint/4X/270p_qp28/g_rtsrgan_4x/ --scale_factor 4

python generate_model_to_pb.py --model evsrnet --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path checkpoint/4X/270p_qp28/evsrnet_4x/ --scale_factor 4

python generate_model_to_pb.py --model rtvsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/4X/270p_qp28/g_rtvsrgan_4x/ --scale_factor 4



