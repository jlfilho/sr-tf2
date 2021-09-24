import tensorflow as tf
from dataset import Dataset
import argparse
from model_espcn import espcn 
from model_rtvsrgan import rtvsrgan
from model_discriminator import discriminator
from model_gan import GAN
from metrics import psnr, ssim
from save_img_callback import SaveImageCallback


MODEL='rtvsrgan'
BATCH_SIZE = 64
TEST_BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 45150
OPTIMIZER='adam'
LEARNING_RATE = 1e-3
LEARNING_DECAY_RATE = 1e-1
LEARNING_DECAY_EPOCHS = 40
MOMENTUM = 0.9
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 500
SAVE_NUM = 2
STEPS_PER_LOG = 1000
EPOCHS_PER_SAVE = 10
LOGDIR = 'logdir'

TEST_LOGDIR='test_logdir/'


TRAINING_DATASET_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_football-qp27/dataset.tfrecords'
TRAINING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_football-qp27/dataset_info.txt'

TESTING_DATASET_PATH='/home/joao/Documentos/projetos/ssd/dataset/test_football-qp27/dataset.tfrecords'
TESTING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/ssd/dataset/test_football-qp27/dataset_info.txt'


def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['espcn','rtvsrgan','gan'],
                        help='What model to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--valid_batch_size', type=int, default=TEST_BATCH_SIZE,
                        help='Number of images in test batch')
    parser.add_argument('--train_dataset_path', type=str, default=TRAINING_DATASET_PATH,
                        help='Path to the train dataset')
    parser.add_argument('--train_dataset_info_path', type=str, default=TRAINING_DATASET_INFO_PATH,
                        help='Path to the train dataset info')
    parser.add_argument('--valid_dataset_path', type=str, default=TESTING_DATASET_PATH,
                        help='Path to the test dataset')
    parser.add_argument('--valid_dataset_info_path', type=str, default=TESTING_DATASET_INFO_PATH,
                        help='Path to the train dataset info')
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
    parser.add_argument('--test_logdir', type=str, default=TEST_LOGDIR,
                        help='Where to save tests images')

    return parser.parse_args()

def main():
    args = get_arguments()

    train_dataset = Dataset(args.batch_size,
        args.train_dataset_path,
        args.train_dataset_info_path,
        args.shuffle_buffer_size)

    steps_per_epoch = train_dataset.examples_num // args.batch_size \
        if train_dataset.examples_num % args.batch_size != 0 else 0
    steps_per_epoch = 100

    train_dataset = train_dataset.get_data(args.num_epochs)
    train_batch = train_dataset.map(lambda x0,x1,x2,y: (x1/255.0,y/255.0))

    valid_dataset = Dataset(args.valid_batch_size,
        args.valid_dataset_path,
        args.valid_dataset_info_path)
    
    valid_dataset = valid_dataset.get_data()
    valid_batch = valid_dataset.map(lambda x0,x1,x2,y: (x1/255.0,y/255.0))
    valid_batch = iter(valid_batch).get_next()

    test_batch = valid_dataset.map(lambda x0,x1,x2,y: (x1,y))
    test_batch = iter(test_batch).get_next() 
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path,
        save_weights_only=True,
        monitor='psnr',
        save_freq= 'epoch', 
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.logdir+"/"+args.model,
        histogram_freq=0, 
        write_graph=True,
        write_images=True, 
        write_steps_per_second=True,
        update_freq='epoch')
    
    if args.model == 'espcn':
        model = espcn()
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim])
        if args.load_weights:
            print("Loading weights...")
            model.load_weights(args.ckpt_path)
        
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=model,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback]

        model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch,validation_data=valid_batch)
    
    if args.model == 'rtvsrgan':
        model = rtvsrgan()
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim])
        if args.load_weights:
            print("Loading weights...")
            model.load_weights(args.ckpt_path)
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=model,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback]
        model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch,validation_data=valid_batch)

    if args.model == 'gan':
        g=rtvsrgan()
        d=discriminator()
        gan = GAN(discriminator=d, generator=g)
        gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=[psnr,ssim])
        if (args.load_weights):
            print("Loading weights...")
            gan.load_weights_gen(args.ckpt_path)
        
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=g,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback]

        gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch)
    else:
        exit(1)

if __name__ == '__main__':
    main()