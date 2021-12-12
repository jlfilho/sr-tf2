import tensorflow as tf
import argparse
import numpy as np
import sys
import os

from models.dataset import Dataset
from models.espcn.model_espcn import ESPCN 
from models.rtsrgan.model_generator import g_rtsrgan 
from models.rtsrgan.model_discriminator import d_rtsrgan
from models.ertsrgan.model_generator import g_ertsrgan
from models.ertsrgan.model_discriminator import d_ertsrgan,rad_ertsrgan
from models.rtsrgan.model_gan import GAN
from models.ertsrgan.model_ragan import RaGAN
from models.metrics import psnr, ssim, rmse
from models.save_img_callback import SaveImageCallback
from models.utils import scale_1 as scale
from models.losses import VGGLossNoActivation as VGGLoss, GANLoss



MODEL='rtvsrgan'
BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 64
OPTIMIZER='adam'
TYPE_REDUCE_LR='OnPlateau'
LEARNING_RATE = 1e-4
LEARNING_DECAY_RATE = 1e-1
LEARNING_DECAY_EPOCHS = 20
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 100
VAL_STEPS = 1
TEST_STEPS = 0
EPOCHS_PER_SAVE = 5
LOGDIR = 'logdir'
CHECKPOINT = 'checkpoint/'
TRAINNABLE_LAYER = 'conv1'
PATH_TO_EVAL = 'test_logdir/stats.txt'
TEST_LOGDIR='test_logdir/'

HOT_TEST_SIZE=5
LR_HOT_TEST_PATH="datasets/loaded_harmonic/img_test/lr/270p_qp28/"
HR_HOT_TEST_PATH="datasets/loaded_harmonic/img_test/hr/1080p/"

TRAIN_DATASET_PATH='datasets/loaded_harmonic/output/train/2X/270p_qp17/dataset.tfrecords'
TRAIN_DATASET_INFO_PATH='datasets/loaded_harmonic/output/train/2X/270p_qp17/dataset_info.txt'

VAL_DATASET_PATH='datasets/loaded_harmonic/output/val/2X/270p_qp17/dataset.tfrecords'
VAL_DATASET_INFO_PATH='datasets/loaded_harmonic/output/val/2X/270p_qp17/dataset_info.txt'

TEST_DATASET_PATH='datasets/loaded_harmonic/output/test/2X/270p_qp17/dataset.tfrecords'
TEST_DATASET_INFO_PATH='datasets/loaded_harmonic/output/test/2X/270p_qp17/dataset_info.txt'


def get_arguments():
    parser = argparse.ArgumentParser(description='train one of the models for image and video super-resolution')
    parser.add_argument('--model', type=str, default=MODEL, choices=['espcn','g_rtsrgan','rtsrgan','g_ertsrgan','ertsrgan'],
                        help='What model to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch')
    parser.add_argument('--train_dataset_path', type=str, default=TRAIN_DATASET_PATH,
                        help='Path to the train dataset')
    parser.add_argument('--train_dataset_info_path', type=str, default=TRAIN_DATASET_INFO_PATH,
                        help='Path to the train dataset info')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=STEPS_PER_EPOCH, 
                        help='Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.')

    parser.add_argument('--val_batch_size', type=int, default=VAL_BATCH_SIZE,
                        help='Number of images in val batch')
    parser.add_argument('--val_dataset_path', type=str, default=VAL_DATASET_PATH,
                        help='Path to the val dataset')
    parser.add_argument('--val_dataset_info_path', type=str, default=VAL_DATASET_INFO_PATH,
                        help='Path to the val dataset info')
    parser.add_argument('--validation_steps', type=int, default=VAL_STEPS, 
                        help='Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.')
    
    parser.add_argument('--test_batch_size', type=int, default=TEST_BATCH_SIZE,
                        help='Number of images in test batch')
    parser.add_argument('--test_dataset_path', type=str, default=TEST_DATASET_PATH,
                        help='Path to the test dataset')
    parser.add_argument('--test_dataset_info_path', type=str, default=TEST_DATASET_INFO_PATH,
                        help='Path to the test dataset info')
    parser.add_argument('--test_steps', type=int, default=TEST_STEPS, 
                        help='Total number of steps (batches of samples) to draw before stopping when performing evaluate at the end of every epoch.')

    parser.add_argument('--hot_test_size', type=int, default=HOT_TEST_SIZE,
                        help='Number of images in hot test')
    parser.add_argument('--lr_hot_test_path', type=str, default=LR_HOT_TEST_PATH,
                        help='Path to the hot test dataset')
    parser.add_argument('--hr_hot_test_path', type=str, default=HR_HOT_TEST_PATH,
                        help='Path to the hr hot test path')

    parser.add_argument('--ckpt_path', default=CHECKPOINT,
                        help='Path to the model checkpoint to evaluate')
    parser.add_argument('--load_weights', action='store_true',
                        help='Load weights')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='Transfer learning from lower-upscale model')
    parser.add_argument('--trainable_layer', type=str, default=TRAINNABLE_LAYER,
                        help='Transfer learning from lower-upscale model')
    parser.add_argument('--scaleFrom', type=int, default=2,
                        help='Perform transfer learning from lower-upscale model' )
    parser.add_argument('--shuffle_buffer_size', type=int, default=SHUFFLE_BUFFER_SIZE,
                        help='Buffer size used for shuffling examples in dataset')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate used for training')
    parser.add_argument('--lr_decay_rate', type=float, default=LEARNING_DECAY_RATE,
                        help='Learning rate decay rate used in exponential decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=LEARNING_DECAY_EPOCHS,
                        help='Number of epochs before full decay rate tick used in exponential decay')

    parser.add_argument('--type_reduce_lr', type=str, default=TYPE_REDUCE_LR, choices=['plateau','schedules'],
                        help='Type of reduce learning rate')
    
    parser.add_argument('--epochs_per_save', type=int, default=EPOCHS_PER_SAVE,
                        help='How often to save checkpoints')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Where to save checkpoints and summaries')
    parser.add_argument('--test_logdir', type=str, default=TEST_LOGDIR,
                        help='Where to save tests images')
    
    parser.add_argument('--path_to_eval', type=str, default=PATH_TO_EVAL,
                        help='Path to save evals')
    return parser.parse_args()


def main():
    
    args = get_arguments()
    # train dataset
    train_dataset = Dataset(args.batch_size,
        args.train_dataset_path,
        args.train_dataset_info_path,
        args.shuffle_buffer_size)
    
    scale_factor = train_dataset.scale_factor

    if args.steps_per_epoch == 0:
        steps_per_epoch = train_dataset.examples_num // args.batch_size \
            if train_dataset.examples_num % args.batch_size != 0 else 0
    else:
        steps_per_epoch = args.steps_per_epoch
    

    train_dataset = train_dataset.get_data(args.num_epochs)
    train_batch = train_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))
    
    # val dataset
    val_dataset = Dataset(args.val_batch_size,
        args.val_dataset_path,
        args.val_dataset_info_path,
        args.shuffle_buffer_size)

    if args.validation_steps == 0:
        validation_steps = val_dataset.examples_num // args.val_batch_size \
            if val_dataset.examples_num % args.val_batch_size != 0 else 0
    else:
        validation_steps = args.validation_steps
    
    val_dataset = val_dataset.get_data()
    val_batch = val_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))
    
    # test dataset
    test_dataset = Dataset(args.test_batch_size,
        args.test_dataset_path,
        args.test_dataset_info_path,
        args.shuffle_buffer_size)

    if args.test_steps == 0:
        test_steps = test_dataset.examples_num // args.test_batch_size \
            if test_dataset.examples_num % args.test_batch_size != 0 else 0
    else:
        test_steps = args.test_steps
    
    test_dataset = test_dataset.get_data()
    test_batch = test_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))


    # hot test
    lr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(args.lr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
    hr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(args.hr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
    test_print = [lr_img_paths,hr_img_paths]
    
    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
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
            patience=100, verbose=1,
            mode='min', 
            restore_best_weights=True)
    
    if args.type_reduce_lr == 'plateau':
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=args.lr_decay_rate,
                                        patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)
    if args.type_reduce_lr == 'schedules':
        def scheduler(epoch, lr):
            if epoch in [100,150]:
                return lr * tf.math.exp(-0.1)
            else:
                return lr

        reduce_lr=tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    if args.model == 'espcn':    
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval=train_espcn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)
        
        print_eval(args.path_to_eval,eval,args.model)
    
    if args.model == 'g_rtsrgan':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval=train_g_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model)

    if args.model == 'g_ertsrgan':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval=train_g_ertsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model)

    if args.model == 'rtsrgan':
        callbacks=[tensorboard_callback]
        eval=train_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model)

    if args.model == 'ertsrgan':
        callbacks=[tensorboard_callback]
        eval=train_ertsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model)
       
    else:
        exit(1)


def trainable_weights(model):
    print("Weights:", len(model.weights))
    print("Trainable_weights:", len(model.trainable_weights))
    print("Non_trainable_weights:", len(model.non_trainable_weights))


def print_eval(file_stats,eval,model_name):
    sys.stdout=open(file_stats,"a")
    print(model_name)
    print (eval)
    sys.stdout.close()

def train_espcn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,
                file_writer_cm,trainable_layer):
    #model = espcn(scale_factor=scale_factor)
    model = ESPCN(scale_factor=scale_factor)
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = espcn(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
    trainable_weights(model)

    save_img_callback = SaveImageCallback(
        model=model,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)

    callbacks.append(save_img_callback)

    model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
    verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)

    print("Evaluate model")
    eval = model.evaluate(test_batch, verbose=1, steps=test_steps)
    return eval

def train_g_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,
                file_writer_cm,trainable_layer):
    model = g_rtsrgan(scale_factor=scale_factor)
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = g_rtsrgan(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
    trainable_weights(model)

    save_img_callback = SaveImageCallback(
        model=model,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)

    callbacks.append(save_img_callback)

    model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
    verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)

    print("Evaluate model")
    eval = model.evaluate(test_batch, verbose=1, steps=test_steps)
    return eval


def train_g_ertsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    model = g_ertsrgan(scale_factor=scale_factor)
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = g_ertsrgan(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
    loss = tf.keras.losses.MeanSquaredError()
    #loss = tf.keras.losses.MeanAbsoluteError()
    #loss = tf.keras.losses.Huber()
    model.compile(optimizer=opt, loss=loss, metrics=[psnr,ssim,rmse])
    trainable_weights(model)
   
    save_img_callback = SaveImageCallback(
        model=model,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)

    callbacks.append(save_img_callback)

    model.fit(train_batch,epochs=args.num_epochs,callbacks=callbacks,
    verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)

    print("Evaluate model")
    eval = model.evaluate(test_batch, verbose=1, steps=test_steps)
    return eval


def train_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    g=g_rtsrgan(scale_factor=scale_factor)
    d=d_rtsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
    gan = GAN(discriminator = d, generator = g)
    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr)
    #cont_loss = tf.keras.losses.MeanAbsoluteError()
    #cont_loss = tf.keras.losses.Huber()
    cont_loss = tf.keras.losses.MeanSquaredError()
    perc_loss = vgg_loss.perceptual_loss
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lbd = 1 * 1e-5
    eta = 1 * 1e-2
    mu = 1 * 1e-2
    gan_loss=GANLoss(perc_loss, cont_loss, adv_loss,lbd,eta,mu)
        
    if (args.load_weights):
        print("Loading weights...")
        checkpoint_paph="{}g_rtsrgan_{}x/model.ckpt".format(args.ckpt_path,scale_factor) 
        gan.load_weights_gen(checkpoint_paph)
        for i in range(len(g.layers)):
            if(g.layers[i].name == trainable_layer):
                break
            else:
                g.layers[i].trainable=False
    
    gan.compile(d_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                g_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                d_loss = gan_loss.discriminator_loss,
                g_loss = gan_loss.generator_loss,
                metrics=[psnr,ssim,rmse])
    trainable_weights(gan)

    save_img_callback = SaveImageCallback(
        model=g,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)

    callbacks.append(save_img_callback)

    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='rmse', 
        min_delta=1e-5,
        patience=50, verbose=1,
        mode='min', 
        restore_best_weights=True)
    callbacks.append(earlystopping)

    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='rmse',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='rmse', factor=args.lr_decay_rate,patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)
    callbacks.append(reduce_lr) 

    gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=steps_per_epoch)
    checkpoint_paph="{}{}_{}x/g_rtsrgan/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    gan.save_weights_gen(checkpoint_paph)


    print("Evaluate model")
    eval = g.evaluate(test_batch, verbose=1, steps=test_steps)
    return eval




def train_ertsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    g=g_ertsrgan(scale_factor=scale_factor)
    d=d_ertsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
    ra_d=rad_ertsrgan(discriminator=d,shape_hr=(36*scale_factor,36*scale_factor,1))
    ra_gan = RaGAN(ra_discriminator=ra_d, generator=g)
    

    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr)
    #cont_loss = tf.keras.losses.MeanAbsoluteError()
    #cont_loss = tf.keras.losses.Huber()
    cont_loss = tf.keras.losses.MeanSquaredError()
    perc_loss = vgg_loss.perceptual_loss
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lbd = 1 * 1e-5
    eta = 1 * 1e-2
    mu = 1 * 1e-2
    gan_loss=GANLoss(perc_loss, cont_loss, adv_loss,lbd,eta,mu)


    ra_gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                ra_d_loss=gan_loss.discriminator_loss,
                g_loss = gan_loss.generator_loss,
                metrics=[psnr,ssim,rmse])
    trainable_weights(ra_gan)

    if (args.load_weights):
        print("Loading weights...")
        checkpoint_paph="{}g_ertsrgan_{}x/model.ckpt".format(args.ckpt_path,scale_factor) 
        ra_gan.load_weights_gen(checkpoint_paph)
        for i in range(len(g.layers)):
            if(g.layers[i].name == trainable_layer):
                break
            else:
                g.layers[i].trainable=False

    save_img_callback = SaveImageCallback(
        model=g,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)    
    callbacks.append(save_img_callback)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='rmse', factor=args.lr_decay_rate, patience=args.lr_decay_epochs, mode='min', min_lr=1e-6,verbose=1)
    callbacks.append(reduce_lr)

    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='rmse', 
        min_delta=1e-5,
        patience=50, verbose=1,
        mode='min', 
        restore_best_weights=True)
    callbacks.append(earlystopping)

    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='rmse',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    ra_gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)
    checkpoint_paph="{}{}_{}x/g_ertsrgan/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    ra_gan.save_weights_gen(checkpoint_paph)

    print("Evaluate model")
    eval = g.evaluate(test_batch, verbose=1, steps=test_steps)
    return eval


if __name__ == '__main__':
    main()
