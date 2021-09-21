import tensorflow as tf
from dataset import Dataset
import argparse
from model_espcn import espcn 
from model_rtvsrgan import rtvsrgan
from model_gan import GAN
from metrics import psnr, ssim
from save_img_callback import SaveImageCallback


MODEL='espcn'
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 45150
OPTIMIZER='adam'
LEARNING_RATE = 1e-3
LEARNING_DECAY_RATE = 1e-1
LEARNING_DECAY_EPOCHS = 40
MOMENTUM = 0.9
NUM_EPOCHS = 1000
STEPS_PER_EPOCH = 500
SAVE_NUM = 2
STEPS_PER_LOG = 1000
EPOCHS_PER_SAVE = 5
LOGDIR = 'logdir'

TRAINING_LOGDIR='logdir'
EVAL_LOGDIR='logdir/espcn_batch_32_lr_1e-3_decay_adam/test'


TRAINING_DATASET_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_football-qp17/dataset.tfrecords'
TRAINING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_football-qp17/dataset_info.txt'

TESTING_DATASET_PATH='/home/joao/Documentos/projetos/sr/datasets/test_football-qp17/dataset.tfrecords'
TESTING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/sr/datasets/test_football-qp17/dataset_info.txt'


def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['espcn','rtvsrgan'],
                        help='What model to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--dataset_path', type=str, default=TRAINING_DATASET_PATH,
                        help='Path to the dataset')
    parser.add_argument('--dataset_info_path', type=str, default=TRAINING_DATASET_INFO_PATH,
                        help='Path to the dataset info')
    parser.add_argument('--ckpt_path', default=LOGDIR+'/model.ckpt',
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--load_weights', action='store_true',
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--shuffle_buffer_size', type=int, default=SHUFFLE_BUFFER_SIZE,
                        help='Buffer size used for shuffling examples in dataset')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, choices=['adam', 'momentum', 'sgd'],
                        help='What optimizer to use for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--use_lr_decay', action='store_true',
                        help='Whether to apply exponential decay to the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=LEARNING_DECAY_RATE,
                        help='Learning rate decay rate used in exponential decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=LEARNING_DECAY_EPOCHS,
                        help='Number of epochs before full decay rate tick used in exponential decay')
    parser.add_argument('--staircase_lr_decay', action='store_true',
                        help='Whether to decay the learning rate at discrete intervals')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epochs', type=int, default=STEPS_PER_EPOCH,
                        help='How many steps per epochs')
    parser.add_argument('--save_num', type=int, default=SAVE_NUM,
                        help='How many images to write to summary')
    parser.add_argument('--steps_per_log', type=int, default=STEPS_PER_LOG,
                        help='How often to save summaries')
    parser.add_argument('--epochs_per_save', type=int, default=EPOCHS_PER_SAVE,
                        help='How often to save checkpoints')
    parser.add_argument('--use_mc', action='store_true',
                        help='Whether to use motion compensation in video super resolution model')
    parser.add_argument('--mc_independent', action='store_true',
                        help='Whether to train motion compensation network independent from super resolution network')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save checkpoints and summaries')

    return parser.parse_args()


def main():
    args = get_arguments()

    if args.model == 'espcn':
        model = espcn()
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim])
    elif args.model == 'rtvgan':
        model = rtvsrgan()
    else:
        exit(1)
    
    print(args.load_weights)
    if args.load_weights:
        model.load_weights(args.ckpt_path)
    
    dataset = Dataset(args.batch_size,
        args.dataset_path,
        args.dataset_info_path,
        args.shuffle_buffer_size)


    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path,
        save_weights_only=True,
        monitor='psnr',
        save_freq= 'epoch', 
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.logdir,
        histogram_freq=0, 
        write_graph=True,
        write_images=True, 
        write_steps_per_second=False,
        update_freq='epoch')

    save_img__callback = SaveImageCallback(args=args,model=model)
    callbacks=[checkpoint_callback,tensorboard_callback,save_img__callback]

    


    num_steps_in_epoch = dataset.examples_num // args.batch_size + \
        1 if dataset.examples_num % args.batch_size != 0 else 0

    _, iterator = dataset.get_data()
    batch = iterator.next()

    model.fit(tf.keras.layers.Lambda(lambda x: x / 255.0)(batch[1]),tf.keras.layers.Lambda(lambda x: x / 255.0)(batch[3]),
    epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=num_steps_in_epoch, 
    validation_data=(tf.keras.layers.Lambda(lambda x: x / 255.0)(save_img__callback.data_batch['lr1']),
    tf.keras.layers.Lambda(lambda x: x / 255.0)(save_img__callback.data_batch['hr'])))

if __name__ == '__main__':
    main()