import tensorflow as tf
import os
import argparse
from model_espcn import espcn 
from model_rtvsrgan import rtvsrgan


def get_arguments():
    parser = argparse.ArgumentParser(description='generate binary model file')
    parser.add_argument('--model', type=str, default='espcn', choices=['espcn', 'rtvsrgan'],
                        help='What model to use for generation')
    parser.add_argument('--output_folder', type=str, default='./dnn_bin_models/',
                        help='where to put generated files')
    parser.add_argument('--ckpt_path', default=None,
                        help='Path to the model checkpoint, from which weights are loaded')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2, 3, 4],
                        help='What scale factor was used for chosen model')
    return parser.parse_args()


def main():
    args = get_arguments()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    if args.ckpt_path is None:
        print("Path to the checkpoint file was not provided")
        exit(1)

    if args.model == 'rtvsrgan':
        model = rtvsrgan()
        model.load_weights(args.ckpt_path)
    elif args.model == 'espcn':
        model = espcn()
        model.load_weights(args.ckpt_path)
        
    else:
        exit(1)


if __name__ == '__main__':
    main()
