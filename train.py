import tensorflow as tf
import argparse
import numpy as np

from models.dataset import Dataset
from models.model_espcn import espcn 
from models.rtsrgan.model_generator import g_rtsrgan 
from models.rtsrgan.model_discriminator import d_rtsrgan
from models.ertsrgan.model_generator import g_ertsrgan
from models.ertsrgan.model_discriminator import d_ertsrgan,rad_ertsrgan
from models.rtsrgan.model_gan import GAN
from models.ertsrgan.model_ragan import RaGAN
from models.metrics import psnr, ssim, rmse
from models.save_img_callback import SaveImageCallback
from models.losses import discriminator_loss, generator_loss
from models.utils import scale_1 as scale


MODEL='rtvsrgan'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 0
OPTIMIZER='adam'
LEARNING_RATE = 1e-4
LEARNING_DECAY_RATE = 1e-1
LEARNING_DECAY_EPOCHS = 10
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 50
EPOCHS_PER_SAVE = 5
LOGDIR = 'logdir'
CHECKPOINT = 'checkpoint/'

TEST_LOGDIR='test_logdir/'


TRAINING_DATASET_PATH='/home/joao/Documentos/projetos/sr/datasets/train_football-qp20_blur/dataset.tfrecords'
TRAINING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/sr/datasets/train_football-qp20_blur/dataset_info.txt'

TESTING_DATASET_PATH='/home/joao/Documentos/projetos/sr/datasets/test_football-qp20_blur/dataset.tfrecords'
TESTING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/sr/datasets/test_football-qp20_blur/dataset_info.txt'

# TRAINING_DATASET_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_div2k/dataset.tfrecords'
# TRAINING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/ssd/dataset/train_div2k/dataset_info.txt'
# TESTING_DATASET_PATH='/home/joao/Documentos/projetos/ssd/dataset/test_div2k/dataset.tfrecords'
# TESTING_DATASET_INFO_PATH='/home/joao/Documentos/projetos/ssd/dataset/test_div2k/dataset_info.txt'

def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['espcn','g_rtsrgan','rtsrgan','g_ertsrgan','ertsrgan'],
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
    parser.add_argument('--ckpt_path', default=CHECKPOINT,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--load_weights', action='store_true',
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--shuffle_buffer_size', type=int, default=SHUFFLE_BUFFER_SIZE,
                        help='Buffer size used for shuffling examples in dataset')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--lr_decay_rate', type=float, default=LEARNING_DECAY_RATE,
                        help='Learning rate decay rate used in exponential decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=LEARNING_DECAY_EPOCHS,
                        help='Number of epochs before full decay rate tick used in exponential decay')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epochs', type=int, default=STEPS_PER_EPOCH,
                        help='How many steps per epochs')
    parser.add_argument('--epochs_per_save', type=int, default=EPOCHS_PER_SAVE,
                        help='How often to save checkpoints')
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

    scale_factor = train_dataset.scale_factor

    if args.steps_per_epochs == 0:
        steps_per_epoch = train_dataset.examples_num // args.batch_size \
            if train_dataset.examples_num % args.batch_size != 0 else 0
    else:
        steps_per_epoch = args.steps_per_epochs
    print(steps_per_epoch)


    train_dataset = train_dataset.get_data(args.num_epochs)
    train_batch = train_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))

    valid_dataset = Dataset(args.valid_batch_size,
        args.valid_dataset_path,
        args.valid_dataset_info_path)
    
    valid_dataset = valid_dataset.get_data()
    valid_batch = valid_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))
    valid_batch = iter(valid_batch).get_next()

    test_batch = valid_dataset.map(lambda x0,x1,x2,y: (x1,y))
    test_batch = iter(test_batch).get_next() 
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path+args.model+'/model.ckpt',
        save_weights_only=True,
        monitor='val_loss',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.logdir+"/"+args.model,
        histogram_freq=1, 
        write_graph=True,
        write_images=True, 
        write_steps_per_second=True,
        update_freq='batch') 
    file_writer_cm = tf.summary.create_file_writer(args.logdir+"/"+args.model + '/validation')
    
    earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-5,
            patience=50, verbose=1,
            mode='min', 
            restore_best_weights=True)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=args.lr_decay_rate,
                                    patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)
    
    initial_learning_rate = 1e-2
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.1,
        staircase=True)
    
    if args.model == 'espcn':
        model = espcn(scale_factor=scale_factor)
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
        #opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
        if args.load_weights:
            print("Loading weights...")
            model.load_weights(args.ckpt_path+args.model+'/model.ckpt')
        
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=model,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir,
            file_writer_cm=file_writer_cm)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback,earlystopping,reduce_lr] 

        model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch,validation_data=valid_batch)
    
    if args.model == 'g_rtsrgan':
        model = g_rtsrgan(scale_factor=scale_factor)
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
        if args.load_weights:
            print("Loading weights...")
            model.load_weights(args.ckpt_path+args.model+'/model.ckpt')
        
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=model,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir,
            file_writer_cm=file_writer_cm)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback,earlystopping,reduce_lr]
        model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch,validation_data=valid_batch)
    
    if args.model == 'g_ertsrgan':
        model = g_ertsrgan(scale_factor=scale_factor)
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0) 
        #opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipnorm=1.0)
        loss = tf.keras.losses.MeanSquaredError()
        #loss = tf.keras.losses.MeanAbsoluteError()
        #loss = tf.keras.losses.Huber()
        model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
        if args.load_weights:
            print("Loading weights...")
            model.load_weights(args.ckpt_path+args.model+'/model.ckpt')
        
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=model,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir,
            file_writer_cm=file_writer_cm)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback,earlystopping,reduce_lr]
        model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch,validation_data=valid_batch)

    if args.model == 'rtsrgan':
        g=g_rtsrgan(scale_factor=scale_factor)
        d=d_rtsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
        gan = GAN(discriminator = d, generator = g)
        gan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                    g_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                    d_loss = discriminator_loss,
                    g_loss = generator_loss,
                    metrics=[psnr,ssim,rmse])
        
        if (args.load_weights):
            print("Loading weights...")
            gan.load_weights_gen(args.ckpt_path+'g_rtsrgan/model.ckpt')
            
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=g,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir,
            file_writer_cm=file_writer_cm)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='rmse', factor=args.lr_decay_rate,
                                    patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback,earlystopping,reduce_lr] 

        gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch)

        gan.save_weights_gen(args.ckpt_path+args.model+'/g_rtsrgan/model.ckpt')

    if args.model == 'ertsrgan':
        g=g_ertsrgan(scale_factor=scale_factor)
        d=d_ertsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
        ra_d=rad_ertsrgan(discriminator=d,shape_hr=(36*scale_factor,36*scale_factor,1))
        ra_gan = RaGAN(ra_discriminator=ra_d, generator=g)
        ra_gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                    g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                    ra_d_loss=discriminator_loss,
                    g_loss = generator_loss,
                    metrics=[psnr,ssim,rmse])

        if (args.load_weights):
            print("Loading weights...")
            ra_gan.load_weights_gen(args.ckpt_path+'g_ertsrgan/model.ckpt')
            
        save_img_callback = SaveImageCallback(
            dataset=test_batch,
            model=g,
            model_name=args.model,
            epochs_per_save=args.epochs_per_save,
            log_dir=args.test_logdir,
            file_writer_cm=file_writer_cm)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='rmse', factor=args.lr_decay_rate,
                                    patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)

        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='rmse', 
            min_delta=1e-5,
            patience=50, verbose=1,
            mode='min', 
            restore_best_weights=True)
        
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.ckpt_path+args.model+'/model.ckpt',
            save_weights_only=True,
            monitor='rmse',
            save_freq= 'epoch', 
            mode='min',
            save_best_only=True)

        callbacks=[checkpoint_callback,tensorboard_callback,save_img_callback,earlystopping,reduce_lr] 

        ra_gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,
        verbose=1,steps_per_epoch=steps_per_epoch)

        ra_gan.save_weights_gen(args.ckpt_path+args.model+'/g_rtvsrgan/model.ckpt')
    else:
        exit(1)


if __name__ == '__main__':
    main()
