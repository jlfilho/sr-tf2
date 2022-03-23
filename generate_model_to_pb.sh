#!/bin/bash

# python generate_model_to_pb.py --model rtvsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x2/ --ckpt_path ./checkpoint/g_rtvsrgan_2x/ --scale_factor 2

# python generate_model_to_pb.py --model rtvsrgan --output_folder ../ffmpeg-tensorflow/dnn_models/x3/ --ckpt_path ./checkpoint/g_rtvsrgan_3x/ --scale_factor 3

# python generate_model_to_pb.py --model escpn --output_folder dnn_bin_models/ --ckpt_path ./checkpoint/3X/270p_qp17/espcn_3x/ --scale_factor 3

# X4 - generic
python generate_model_to_pb.py --model espcn --cluster generic --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/generic/percsr_4x/espcn/generator/ --scale_factor 4

python generate_model_to_pb.py --model imdn --cluster generic --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/generic/percsr_4x/imdn/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtsrgan --cluster generic --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/generic/percsr_4x/g_rtsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtvsrgan --cluster generic --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/generic/percsr_4x/g_rtvsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model evsrnet --cluster generic --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/generic/percsr_4x/evsrnet/generator/ --scale_factor 4


# X4 - game
python generate_model_to_pb.py --model espcn --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/game/percsr_4x/espcn/generator/ --scale_factor 4

python generate_model_to_pb.py --model imdn --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/game/percsr_4x/imdn/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtsrgan --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/game/percsr_4x/g_rtsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtvsrgan --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/game/percsr_4x/g_rtvsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model evsrnet --cluster game --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/game/percsr_4x/evsrnet/generator/ --scale_factor 4


# X4 - sport
python generate_model_to_pb.py --model espcn --cluster sport --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/sport/percsr_4x/espcn/generator/ --scale_factor 4

python generate_model_to_pb.py --model imdn --cluster sport --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/sport/percsr_4x/imdn/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtsrgan --cluster sport --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/sport/percsr_4x/g_rtsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtvsrgan --cluster sport --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/sport/percsr_4x/g_rtvsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model evsrnet --cluster sport --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/sport/percsr_4x/evsrnet/generator/ --scale_factor 4

# X4 - podcast
python generate_model_to_pb.py --model espcn --cluster podcast --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/podcast/percsr_4x/espcn/generator/ --scale_factor 4

python generate_model_to_pb.py --model imdn --cluster podcast --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/podcast/percsr_4x/imdn/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtsrgan --cluster podcast --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/podcast/percsr_4x/g_rtsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model g_rtvsrgan --cluster podcast --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/podcast/percsr_4x/g_rtvsrgan/generator/ --scale_factor 4

python generate_model_to_pb.py --model evsrnet --cluster podcast --output_folder ../ffmpeg-tensorflow/dnn_models/x4/ --ckpt_path ./checkpoint/tmp/podcast/percsr_4x/evsrnet/generator/ --scale_factor 4


