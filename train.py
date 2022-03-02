import tensorflow as tf
import argparse
import os
import statistics as stat



from models.utils import plot_test_images, plot_images, print_metrics

from models.espcn.model_espcn import ESPCN as espcn

from models.evsrnet.model_evsrnet import EVSRNet

from models.rtsrgan.model_generator import G_RTSRGAN as g_rtsrgan
from models.rtsrgan.model_discriminator import d_rtsrgan
from models.rtsrgan.model_gan import GAN

from models.rtvsrgan.model_generator import G_RTVSRGAN as g_rtvsrgan 
from models.rtvsrgan.KnowledgeDistillation import Distiller

from models.rtvsrgan.model_discriminator import d_rtvsrgan, rad_rtvsrgan
from models.rtvsrgan.model_ragan import RaGAN

from models.rtvsrgan.model_discriminator import d_percsr, rad_percsr
from models.percsr.model_percsr import PercSR
from models.percsr.model_teacher import Teacher


from models.imdn.model_imdn import IMDN

from models.dataset import Dataset
from models.metrics import psnr, ssim, rmse, lpips
from models.losses import VGGLossNoActivation as VGGLoss, GANLoss

from models.save_img_callback import SaveImageCallback
from models.utils import scale_1 as scale


hot_test= {'hot_test_generic': {
  'lr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/generic/lr/270p_qp17/",
  'hr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/generic/hr/1080p/"
},
'hot_test_game': {
  'lr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/game/lr/270p_qp17/",
  'hr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/game/hr/1080p/"
},
'hot_test_sport': {
  'lr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/sport/lr/270p_qp17/",
  'hr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/sport/hr/1080p/"
},
'hot_test_podcast': {
  'lr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/podcast/lr/270p_qp17/",
  'hr_hot_test_path': "datasets/loaded_harmonic/img_hot_test/podcast/hr/1080p/"
}}


test= {
'test_generic': {
  'lr_test_path': "/home/joao/Documentos/projetos/sr-tf2/datasets/loaded_harmonic/img_test/lr/270p_qp17/",
  'hr_test_path': "/home/joao/Documentos/projetos/sr-tf2/datasets/loaded_harmonic/img_test/hr/1080p/",
  'logdir': "test_logdir/test/generic/"
},
'test_game': {
  'lr_test_path': "/media/joao/SAMSUNG/Youtube/game/img_test/lr/270p_qp17/",
  'hr_test_path': "/media/joao/SAMSUNG/Youtube/game/img_test/hr/1080p/",
  'logdir': "test_logdir/test/game/"
},
'test_sport': {
  'lr_test_path': "/media/joao/SAMSUNG/Youtube/sport/img_test/lr/270p_qp17/",
  'hr_test_path': "/media/joao/SAMSUNG/Youtube/sport/img_test/hr/1080p/",
  'logdir': "test_logdir/test/sport/"
},
'test_podcast': {
  'lr_test_path': "/media/joao/SAMSUNG/Youtube/podcast/img_test/lr/270p_qp17/",
  'hr_test_path': "/media/joao/SAMSUNG/Youtube/podcast/img_test/hr/1080p/",
  'logdir': "test_logdir/test/podcast/"
}}


test_datasets = {
'test_generic': {
  'test_dataset_path': "datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset.tfrecords",
  'test_dataset_info_path': "datasets/loaded_harmonic/output/generic/test/4X/270p_qp17/dataset_info.txt"
},
'test_game': {
  'test_dataset_path': "datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset.tfrecords",
  'test_dataset_info_path': "datasets/loaded_harmonic/output/game/test/4X/270p_qp17/dataset_info.txt"
},
'test_sport': {
  'test_dataset_path': "datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset.tfrecords",
  'test_dataset_info_path': "datasets/loaded_harmonic/output/sport/test/4X/270p_qp17/dataset_info.txt"
},
'test_podcast': {
  'test_dataset_path': "datasets/loaded_harmonic/output/podcast/test/4X/270p_qp17/dataset.tfrecords",
  'test_dataset_info_path': "datasets/loaded_harmonic/output/podcast/test/4X/270p_qp17/dataset_info.txt"
}}



LIST_MODEL=['espcn','g_rtsrgan','rtsrgan','g_rtvsrgan','teacher','rtvsrgan','imdn','k_dist','percsr','evsrnet']
MODEL='rtvsrgan'
LIST_GENERATOR=[None,'espcn','g_rtsrgan','imdn','evsrnet','g_rtvsrgan']
GENERATOR=None
BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 64

LIST_TEST_CLUSTER = ['generic','game','sport','podcast']
TEST_CLUSTER = ['sport']

SCHEDULE_VALUES=[100]

# Knowledge distillation model
LOSS_FN='mae'
DISTILLATION_RATE=0.8
ALPHA=0.3
BETA=0.65
LIST_WEIGHTS=[1e-5,1e-2,1e-2]

TYPE_REDUCE_LR='schedules'
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
TRAINNABLE_LAYER = 'final'
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
    parser.add_argument('--model', type=str, default=MODEL, choices=LIST_MODEL,
                        help='What model to train', required=True)
    parser.add_argument('--generator', type=str, default=GENERATOR, choices=LIST_GENERATOR,
                        help='What model to train', required=False)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Number of images in batch', required=True)
    parser.add_argument('--train_dataset_path', type=str, default=TRAIN_DATASET_PATH,
                        help='Path to the train dataset', required=True)
    parser.add_argument('--train_dataset_info_path', type=str, default=TRAIN_DATASET_INFO_PATH,
                        help='Path to the train dataset info', required=True)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs', required=True)
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
    parser.add_argument('--test_cluster', nargs='*', type=str, default=TEST_CLUSTER, choices=LIST_TEST_CLUSTER,
                        help='What cluster dataset to eval', required=False)
         

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
    parser.add_argument('--load_weights_perc', action='store_true',
                        help='Load weights perceptual')
    parser.add_argument('--eval', action='store_true',
                        help='Avaluete model')
    parser.add_argument('--range_to_save', type=int, default=10,
                        help='Range of image to save for teste.' )                    
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
    parser.add_argument('--schedule_values',nargs='*', type=int, default=SCHEDULE_VALUES,
                        help='list of epochs values to reduce lr')

    parser.add_argument('--loss_fn', type=str, default=LOSS_FN, choices=['mse','mae','huber', 'fea'],
                        help='Set the loss function to knowledge distillation model')
    parser.add_argument('--distillation_rate', type=float, default=DISTILLATION_RATE,
                        help='Distillation rate in knowledge distillation model')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='Weight for distillation loss function in knowledge distillation model')
    parser.add_argument('--beta', type=float, default=BETA,
                        help='Weight for perceptual loss function in knowledge distillation model')
    parser.add_argument('--list_weights', nargs='*', type=float, default=LIST_WEIGHTS,
                        help='Auxiliary list to weight values')
    parser.add_argument('--inter_method', type=str, default=None, choices=['bilinear','lanczos3','lanczos5','bicubic','nearest','mitchellcubic'],
                        help='Type of interpolation resize used of same models')

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
    elif args.type_reduce_lr == 'schedules':
        def scheduler(epoch, lr):
            if epoch in args.schedule_values:
                return lr * tf.math.exp(-0.1)
            else:
                return lr
        reduce_lr=tf.keras.callbacks.LearningRateScheduler(scheduler)
    else: 
        print("--type_reduce_lr not valid!")
        exit(1)
    
    if args.model == 'espcn':    
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval,run_time=train_espcn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)
        
        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    elif args.model == 'imdn':    
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval,run_time=train_imdn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)
        
        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    
    elif args.model == 'g_rtsrgan':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval, run_time=train_g_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)

    elif args.model == 'rtsrgan':
        callbacks=[tensorboard_callback]
        eval,run_time=train_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)

    elif args.model == 'evsrnet':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr]
        eval,run_time=train_evsrnet(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)


    # Ours models
    elif args.model == 'g_rtvsrgan':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval,run_time=train_g_rtvsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    
    elif args.model == 'teacher':
        callbacks=[checkpoint_callback,tensorboard_callback,earlystopping,reduce_lr] 
        eval,run_time=train_teacher(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)

    elif args.model == 'rtvsrgan':
        callbacks=[tensorboard_callback,reduce_lr]
        eval,run_time=train_rtvsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)

        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    
    elif args.model == 'k_dist':    
        callbacks=[tensorboard_callback, reduce_lr] 
        eval,run_time=train_k_distillation(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)
        
        print_eval(args.path_to_eval,eval,args.model+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    
    elif args.model == 'percsr':
        callbacks=[tensorboard_callback, reduce_lr] 
        print("CALLING MODEL {}".format(args.model))
        eval,run_time=train_percsr(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer=args.trainable_layer)
        
        print_eval(args.path_to_eval,eval,args.model+'_'+args.generator+"_{}X_q{}".format(str(scale_factor),str(args.train_dataset_path).split('_q')[-1]),run_time)
    else:
        exit(1)


def trainable_weights(model):
    print("Weights:", len(model.weights))
    print("Trainable_weights:", len(model.trainable_weights))
    print("Non_trainable_weights:", len(model.non_trainable_weights))

def trainable_layers(model, trainable_layer):
    for i in range(len(model.layers)):
        if(i+1 == trainable_layer):
            break
        else:
            model.layers[i].trainable=False


def print_eval(file_stats,eval,model_name,run_time):
    statsFile=open(file_stats,"a")
    print(model_name, file = statsFile)
    print(eval, file = statsFile)
    print(run_time, file = statsFile)
    statsFile.close()

def saved_model(model, filepath):
    tf.keras.models.save_model(model, filepath, save_traces=True)

def train_espcn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,
                file_writer_cm,trainable_layer):
    model = espcn(scale_factor=scale_factor)
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

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
    trainable_weights(model)

    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        get_test_dataset(model,scale_factor,args)
        exit(1)

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
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval,model.get_run_time()

def train_imdn(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,
                file_writer_cm,trainable_layer):
    model = IMDN(scale_factor=scale_factor)
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = IMDN(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
    trainable_weights(model)

    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        get_test_dataset(model,scale_factor,args)
        exit(1)

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
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval, model.get_run_time()

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

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
    trainable_weights(model)

    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        get_test_dataset(model,scale_factor,args)
        exit(1)

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
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval,model.get_run_time()

def train_rtsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    g=g_rtsrgan(scale_factor=scale_factor)
    g.compile(metrics=[psnr,ssim,rmse,lpips])
    
    d=d_rtsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
    gan = GAN(discriminator = d, generator = g)

    if args.loss_fn == "mse":
        cont_loss = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        cont_loss = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        cont_loss = tf.keras.losses.MeanAbsoluteError()

    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr,cont_loss)
    perc_loss = vgg_loss.custom_perceptual_loss
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
                metrics=[psnr,ssim,rmse,lpips])
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

    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='val_lpips',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=steps_per_epoch)
    checkpoint_paph="{}{}_{}x/g_rtsrgan/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    gan.save_weights_gen(checkpoint_paph)

    print("Evaluate model")
    eval = g.evaluate(test_batch, verbose=1, steps=test_steps)
    saved_model(g, 'saved_model/{}/'.format(args.model))
    return eval, g.get_run_time()


def train_evsrnet(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,
                file_writer_cm,trainable_layer):
    model = EVSRNet(scale_factor=scale_factor,method=args.inter_method)
    model.build((None, None, None,1))
    #print(model.summary())
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae": # default
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
    trainable_weights(model)

    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        get_test_dataset(model,scale_factor,args)
        exit(1)

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
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval,model.get_run_time()

def train_teacher(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    model = Teacher(channels=1,scale_factor=scale_factor,distillation_rate=args.distillation_rate)
    model.build((None, None, None,1))
    print(model.summary())
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)

    if(args.eval==True):
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        model.compile(metrics=[psnr,ssim,rmse,lpips])
        get_test_dataset(model,scale_factor,args)
        exit(1)

    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = g_rtvsrgan(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    if args.loss_fn == "fea":    
        loss_aux = tf.keras.losses.MeanAbsoluteError()
        shape_hr = (36*scale_factor,36*scale_factor,3)    
        vgg_loss = VGGLoss(shape_hr,loss_aux)
        loss_fn = vgg_loss.custom_perceptual_loss

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
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
    if args.loss_fn == "fea": 
        eval = []
    else:
        eval = model.evaluate(test_batch, verbose=1, steps=test_steps)
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval, model.get_run_time()


def train_g_rtvsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    model = g_rtvsrgan(scale_factor=scale_factor,method=args.inter_method)
    if args.load_weights:
        print("Loading weights...")
        model.load_weights(checkpoint_paph)
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = g_rtvsrgan(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(model.layers[i].name))
                model.layers[i].set_weights(modelFrom.layers[i].get_weights())
                model.layers[i].trainable=False
    
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)

    if args.loss_fn == "mse":
        loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[psnr,ssim,rmse,lpips])
    trainable_weights(model)

    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
        model.load_weights(checkpoint_paph)
        print("Evaluate model")
        get_test_dataset(model,scale_factor,args)
        exit(1)
   
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
    saved_model(model, 'saved_model/{}/'.format(args.model))
    return eval,model.get_run_time()


def train_k_distillation(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    
    opt=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0)
    
    if args.loss_fn == "mse":
        aux_loss_fn = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        aux_loss_fn = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        aux_loss_fn = tf.keras.losses.MeanAbsoluteError()
    
    student_loss_fn = tf.keras.losses.MeanSquaredError()
    distillation_loss_fn= tf.keras.losses.MeanAbsoluteError()    
    
    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr,aux_loss_fn)
    perc_loss = vgg_loss.custom_perceptual_loss
      
    teacher = g_rtvsrgan(channels=1,scale_factor=scale_factor)
    print("Loading teacher weights...")
    weights_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,'g_rtvsrgan',scale_factor)
    teacher.load_weights(weights_paph)
    student = g_rtvsrgan(channels=1,scale_factor=scale_factor) 
    student.build((None, None, None,1))

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,
        metrics=[psnr,ssim,rmse,lpips],
        student_loss_fn=student_loss_fn,
        distillation_loss_fn=distillation_loss_fn,
        perc_loss_fn=perc_loss,
        alpha=args.alpha,
        beta=args.beta
    )
    trainable_weights(student)
    if args.load_weights:
        print("Loading student weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,'g_rtvsrgan',scale_factor)
        student.load_weights(checkpoint_paph)
        trainable_layers(student, len(student.layers)-1)
        trainable_weights(student)
        
    if args.transfer_learning:
        checkpoint_paph_from="{}{}_{}x/model.ckpt".format("checkpoint/",args.model,args.scaleFrom)
        print("Transfer learning from {}x-upscale model...".format(args.scaleFrom))
        modelFrom = student(scale_factor=args.scaleFrom)
        modelFrom.load_weights(checkpoint_paph_from)
        for i in range(len(modelFrom.layers)):
            if(modelFrom.layers[i].name == trainable_layer):
                break
            else:
                print("Set_weights in: {} layer".format(student.layers[i].name))
                student.layers[i].set_weights(modelFrom.layers[i].get_weights())
                student.layers[i].trainable=False
    
   
    save_img_callback = SaveImageCallback(
        model=distiller.student,
        model_name=args.model,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)

    callbacks.append(save_img_callback)

    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_rmse', 
        min_delta=1e-5,
        patience=50, verbose=1,
        mode='min', 
        restore_best_weights=True)
    
    callbacks.append(earlystopping)
 
    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='val_lpips',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    # Distill teacher to student
    distiller.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,
    verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)
    checkpoint_paph="{}{}_{}x/g_rtsrgan/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    student.save_weights(checkpoint_paph)

    print("Evaluate model")
    # Evaluate student on test dataset
    eval = distiller.evaluate(test_batch, verbose=1, steps=test_steps)

    saved_model(distiller.student, 'saved_model/{}/'.format(args.model))
    return eval,distiller.student.get_run_time()


def train_rtvsrgan(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):
    g=g_rtvsrgan(scale_factor=scale_factor)
    g.build((None, None, None,1))

    d=d_rtvsrgan(input_shape=(36*scale_factor,36*scale_factor,1))
    ra_d=rad_rtvsrgan(discriminator=d,shape_hr=(36*scale_factor,36*scale_factor,1))
    

    if args.loss_fn == "mse":
        aux_loss = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        aux_loss = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        aux_loss = tf.keras.losses.MeanAbsoluteError()

    cont_loss = tf.keras.losses.MeanSquaredError()
    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr,aux_loss)
    perc_loss = vgg_loss.custom_perceptual_loss

    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lbd = args.list_weights[0]
    eta = args.list_weights[1]
    mu = args.list_weights[2]
    gan_loss=GANLoss(perc_loss, cont_loss, adv_loss,lbd,eta,mu)

    ra_gan = RaGAN(ra_discriminator=ra_d, generator=g)
    ra_gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                ra_d_loss=gan_loss.discriminator_loss,
                g_loss = gan_loss.generator_loss,
                metrics=[psnr,ssim,rmse,lpips])
    if (args.load_weights):
        print("Loading weights...") 
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,'g_rtvsrgan',scale_factor)
        ra_gan.load_weights_gen(checkpoint_paph)
        trainable_layers(g, len(g.layers)-1)
        trainable_weights(g)
        
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


    checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.model,scale_factor)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='val_lpips',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    ra_gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)
    checkpoint_paph="{}{}_{}x/g_rtvsrgan/model.ckpt".format(args.ckpt_path,args.model,scale_factor) 
    ra_gan.save_weights_gen(checkpoint_paph)

    print("Evaluate model")
    eval = ra_gan.evaluate(test_batch, verbose=1)
    saved_model(ra_gan.generator, 'saved_model/{}/'.format(args.model))
    return eval,ra_gan.student.get_run_time()
    

def model_generator(args=None,scale_factor=None):
    if args.generator== 'espcn':
        model= espcn(scale_factor=scale_factor)
    elif args.generator== 'g_rtsrgan':
        model= g_rtsrgan(scale_factor=scale_factor)
    elif args.generator== 'imdn':
        model= IMDN(scale_factor=scale_factor)
    elif args.generator== 'evsrnet':
        model= EVSRNet(scale_factor=scale_factor,method=args.inter_method)
    elif args.generator== 'g_rtvsrgan':
        model= g_rtvsrgan(scale_factor=scale_factor)
    elif args.generator== 'teacher':
        model = Teacher(channels=1,scale_factor=scale_factor,distillation_rate=args.distillation_rate)
    else:
        exit(1)
    return model



def print_hot_test(lr_hot_test_path,hr_hot_test_path,model=None,model_name=None,args=None,scale_factor=2): 
    time_elapsed = plot_test_images(model,lr_hot_test_path,hr_hot_test_path,
             args.test_logdir,scale_factor=scale_factor,model_name=model_name,epoch=0)
    return time_elapsed
    

def get_test_dataset(model,scale_factor,args):
    bic = True
    if ('generic' in args.test_cluster): 
        # test dataset
        test_dataset_path=test_datasets['test_generic']['test_dataset_path']
        test_dataset_info_path=test_datasets['test_generic']['test_dataset_info_path']
        test_dataset = Dataset(
                                args.test_batch_size,
                                test_dataset_path,
                                test_dataset_info_path,
                                args.shuffle_buffer_size)

        if args.test_steps == 0:
            test_steps = test_dataset.examples_num // args.test_batch_size \
                if test_dataset.examples_num % args.test_batch_size != 0 else 0
        else:
            test_steps = args.test_steps
        
        test_dataset = test_dataset.get_data()
        test_batch = test_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))

        name_dataset = args.model+'_'+args.generator+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) if args.generator!=None else args.model+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) 
        print(name_dataset,args.path_to_eval)


        lr_path=test['test_generic']['lr_test_path']
        hr_path=test['test_generic']['hr_test_path']
        logdir=test['test_generic']['logdir']
        lr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_path) if len(filenames)!=0][0])
        hr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_path) if len(filenames)!=0][0])
        if (bic):
            print_metrics(lr_paths, hr_paths, scale_factor=scale_factor)
            exit(1)
        # plot_images("bi", lr_paths, hr_paths, args, logdir+"/"+"bicubic"+"/",scale_factor=scale_factor)
        # plot_images("hr", lr_paths, hr_paths, args, logdir+"/"+"hr"+"/",scale_factor=scale_factor)
        # run_time = plot_images(model, lr_paths, hr_paths, args, logdir+"/"+args.generator+"/",scale_factor=scale_factor)
        run_time = print_hot_test(lr_paths,hr_paths,model=model,model_name=args.model,args=args,scale_factor=scale_factor)
        eval = model.evaluate(test_batch, verbose=1)

        lr_hot_test_path=hot_test['hot_test_generic']['lr_hot_test_path']
        hr_hot_test_path=hot_test['hot_test_generic']['hr_hot_test_path']
        lr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        hr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        test_print = [lr_img_paths,hr_img_paths]

        name_model = "generic"+'_'+args.model+'_'+args.generator if args.generator != None else "generic"+'_'+args.model 
        # run_time = print_hot_test(test_print[0],test_print[1],model=model,model_name=name_model,args=args,scale_factor=scale_factor)
        print_eval(args.path_to_eval,eval,name_dataset,stat.mean(run_time))

    if ('game' in args.test_cluster):
        # test dataset
        test_dataset_path=test_datasets['test_game']['test_dataset_path']
        test_dataset_info_path=test_datasets['test_game']['test_dataset_info_path']
        test_dataset = Dataset(
                                args.test_batch_size,
                                test_dataset_path,
                                test_dataset_info_path,
                                args.shuffle_buffer_size)

        if args.test_steps == 0:
            test_steps = test_dataset.examples_num // args.test_batch_size \
                if test_dataset.examples_num % args.test_batch_size != 0 else 0
        else:
            test_steps = args.test_steps
        
        test_dataset = test_dataset.get_data()
        test_batch = test_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))

        name_dataset = args.model+'_'+args.generator+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) if args.generator != None else args.model+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) 
        print(name_dataset,args.path_to_eval)


        lr_path=test['test_game']['lr_test_path']
        hr_path=test['test_game']['hr_test_path']
        logdir=test['test_game']['logdir']
        lr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_path) if len(filenames)!=0][0])
        hr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_path) if len(filenames)!=0][0])
        if (bic):
            print_metrics(lr_paths, hr_paths, scale_factor=scale_factor)
            exit(1)
        # plot_images("bi", lr_paths, hr_paths, args, logdir+"/"+"bicubic"+"/",scale_factor=scale_factor)
        # plot_images("hr", lr_paths, hr_paths, args, logdir+"/"+"hr"+"/",scale_factor=scale_factor)
        # run_time = plot_images(model, lr_paths, hr_paths, args, logdir+"/"+args.generator+"/",scale_factor=scale_factor)
        run_time = print_hot_test(lr_paths,hr_paths,model=model,model_name=args.model,args=args,scale_factor=scale_factor)
        eval = model.evaluate(test_batch, verbose=1)

        lr_hot_test_path=hot_test['hot_test_game']['lr_hot_test_path']
        hr_hot_test_path=hot_test['hot_test_game']['hr_hot_test_path']
        lr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        hr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        test_print = [lr_img_paths,hr_img_paths]

        name_model = "game"+'_'+args.model+'_'+args.generator if args.generator != None else "game"+'_'+args.model 
        # run_time = print_hot_test(test_print[0],test_print[1],model=model,model_name=name_model,args=args,scale_factor=scale_factor)
        print_eval(args.path_to_eval,eval,name_dataset,stat.mean(run_time))
    
    if ('sport' in args.test_cluster):
        # test dataset
        test_dataset_path=test_datasets['test_sport']['test_dataset_path']
        test_dataset_info_path=test_datasets['test_sport']['test_dataset_info_path']
        test_dataset = Dataset(
                                args.test_batch_size,
                                test_dataset_path,
                                test_dataset_info_path,
                                args.shuffle_buffer_size)

        if args.test_steps == 0:
            test_steps = test_dataset.examples_num // args.test_batch_size \
                if test_dataset.examples_num % args.test_batch_size != 0 else 0
        else:
            test_steps = args.test_steps
        
        test_dataset = test_dataset.get_data()
        test_batch = test_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))

        name_dataset = args.model+'_'+args.generator+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) if args.generator != None else args.model+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) 

        print(name_dataset,args.path_to_eval)


        lr_path=test['test_sport']['lr_test_path']
        hr_path=test['test_sport']['hr_test_path']
        logdir=test['test_sport']['logdir']
        lr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_path) if len(filenames)!=0][0])
        hr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_path) if len(filenames)!=0][0])
        if (bic):
            print_metrics(lr_paths, hr_paths, scale_factor=scale_factor)
            exit(1)
        # plot_images("bi", lr_paths, hr_paths, args, logdir+"/"+"bicubic"+"/",scale_factor=scale_factor)
        # plot_images("hr", lr_paths, hr_paths, args, logdir+"/"+"hr"+"/",scale_factor=scale_factor)
        # run_time = plot_images(model, lr_paths, hr_paths, args, logdir+"/"+args.generator+"/",scale_factor=scale_factor)
        run_time = print_hot_test(lr_paths,hr_paths,model=model,model_name=args.model,args=args,scale_factor=scale_factor)
        eval = model.evaluate(test_batch, verbose=1)

        lr_hot_test_path=hot_test['hot_test_sport']['lr_hot_test_path']
        hr_hot_test_path=hot_test['hot_test_sport']['hr_hot_test_path']
        lr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        hr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        test_print = [lr_img_paths,hr_img_paths]

        name_model = "sport"+'_'+args.model+'_'+args.generator if args.generator != None else "sport"+'_'+args.model 
        # run_time = print_hot_test(test_print[0],test_print[1],model=model,model_name=name_model,args=args,scale_factor=scale_factor)
        print_eval(args.path_to_eval,eval,name_dataset,stat.mean(run_time))

    if ('podcast' in args.test_cluster):
        # test dataset
        test_dataset_path=test_datasets['test_podcast']['test_dataset_path']
        test_dataset_info_path=test_datasets['test_podcast']['test_dataset_info_path']
        test_dataset = Dataset(
                                args.test_batch_size,
                                test_dataset_path,
                                test_dataset_info_path,
                                args.shuffle_buffer_size)

        if args.test_steps == 0:
            test_steps = test_dataset.examples_num // args.test_batch_size \
                if test_dataset.examples_num % args.test_batch_size != 0 else 0
        else:
            test_steps = args.test_steps
        
        test_dataset = test_dataset.get_data()
        test_batch = test_dataset.map(lambda x0,x1,x2,y: (scale(x1),scale(y)))

        name_dataset = args.model+'_'+args.generator+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) if args.generator != None else args.model+"_{}_{}X_q{}".format(str(test_dataset_path).split('/')[3],str(scale_factor),str(test_dataset_path).split('_q')[-1]) 
        print(name_dataset,args.path_to_eval)


        lr_path=test['test_podcast']['lr_test_path']
        hr_path=test['test_podcast']['hr_test_path']
        logdir=test['test_podcast']['logdir']
        lr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_path) if len(filenames)!=0][0])
        hr_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_path) if len(filenames)!=0][0])
        if (bic):
            print_metrics(lr_paths, hr_paths, scale_factor=scale_factor)
            exit(1)
        # plot_images("bi", lr_paths, hr_paths, args, logdir+"/"+"bicubic"+"/",scale_factor=scale_factor)
        # plot_images("hr", lr_paths, hr_paths, args, logdir+"/"+"hr"+"/",scale_factor=scale_factor)
        # run_time = plot_images(model, lr_paths, hr_paths, args, logdir+"/"+args.generator+"/",scale_factor=scale_factor)
        run_time = print_hot_test(lr_paths,hr_paths,model=model,model_name=args.model,args=args,scale_factor=scale_factor)
        eval = model.evaluate(test_batch, verbose=1)

        lr_hot_test_path=hot_test['hot_test_podcast']['lr_hot_test_path']
        hr_hot_test_path=hot_test['hot_test_podcast']['hr_hot_test_path']
        lr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(lr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        hr_img_paths=sorted([[dp+filename for filename in filenames] for dp, dn, filenames in os.walk(hr_hot_test_path) if len(filenames)!=0][0])[0:args.hot_test_size]
        test_print = [lr_img_paths,hr_img_paths]

        name_model = "podcast"+'_'+args.model+'_'+args.generator if args.generator != None else "podcast"+'_'+args.model 
        # run_time = print_hot_test(test_print[0],test_print[1],model=model,model_name=name_model,args=args,scale_factor=scale_factor)
        print_eval(args.path_to_eval,eval,name_dataset,stat.mean(run_time))



def train_percsr(train_batch,steps_per_epoch, validation_steps,val_batch, test_batch, test_steps, test_print, scale_factor,args,callbacks,checkpoint_paph,file_writer_cm,trainable_layer):

    g=model_generator(scale_factor=scale_factor,args=args)
    g.build((None, None, None,1))

    d=d_percsr(input_shape=(36*scale_factor,36*scale_factor,1))
    ra_d=rad_percsr(discriminator=d,shape_hr=(36*scale_factor,36*scale_factor,1))

    if args.loss_fn == "mse":
        aux_loss = tf.keras.losses.MeanSquaredError()        
    if args.loss_fn == "huber":
        aux_loss = tf.keras.losses.Huber()
    if args.loss_fn == "mae":
        aux_loss = tf.keras.losses.MeanAbsoluteError()

    loss_pix = tf.keras.losses.MeanSquaredError()
    shape_hr = (36*scale_factor,36*scale_factor,3)    
    vgg_loss = VGGLoss(shape_hr,aux_loss)
    loss_fea = vgg_loss.custom_perceptual_loss
    loss_dis = tf.keras.losses.MeanAbsoluteError()
    adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    alfa = args.list_weights[0]
    eta = args.list_weights[1]
    lbd = args.list_weights[2]
    mu = args.list_weights[3]

    gan_loss=GANLoss(loss_pix, loss_fea, loss_dis, adv_loss, alfa, eta, lbd, mu)

       
    teacher = Teacher(channels=1,scale_factor=scale_factor,distillation_rate=args.distillation_rate)
    print("Loading teacher weights...")
    weights_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,'teacher',scale_factor)
    teacher.load_weights(weights_paph)
    teacher.build((None, None, None,1))


    ra_gan = PercSR(ra_discriminator=ra_d, generator=g,teacher=teacher)
    ra_gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                   g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate,clipnorm=1.0),
                   perc_loss=gan_loss.generative_loss,
                   metrics=[psnr,ssim,rmse,lpips])
    
    if(args.eval==True):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/{}/model.ckpt".format(args.ckpt_path,args.model,scale_factor,args.generator)
        ra_gan.load_weights(checkpoint_paph)
        print("Evaluate model")
        g.compile(metrics=[psnr,ssim,rmse,lpips])
        get_test_dataset(g,scale_factor,args)
        exit(1)

    if (args.load_weights):
        print("Loading weights...")
        checkpoint_paph="{}{}_{}x/model.ckpt".format(args.ckpt_path,args.generator,scale_factor) 
        ra_gan.load_weights_gen(checkpoint_paph)
        # trainable_layers(g, len(g.layers)-1)
        trainable_weights(g)

    if (args.load_weights_perc):
        print("Loading weights perceptual...")
        checkpoint_paph="{}{}_{}x/{}/model.ckpt".format(args.ckpt_path,args.model,scale_factor,args.generator) 
        ra_gan.load_weights(checkpoint_paph)

        for i in range(len(g.layers)):
            print("Camada: {}".format(g.layers[i].name))
            if(g.layers[i].name == trainable_layer):
                break
            else:
                g.layers[i].trainable=False
        #trainable_layers(g, len(g.layers)-1)
        trainable_weights(g)

        


    save_img_callback = SaveImageCallback(
        model=g,
        model_name=args.model+'_'+args.generator,
        scale_factor=scale_factor,
        epochs_per_save=args.epochs_per_save,
        lr_paths=test_print[0],
        hr_paths=test_print[1],
        log_dir=args.test_logdir,
        file_writer_cm=file_writer_cm)    
    callbacks.append(save_img_callback)


    checkpoint_paph="{}{}_{}x/{}/model.ckpt".format(args.ckpt_path,args.model,scale_factor,args.generator)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_paph,
        save_weights_only=True,
        monitor='val_lpips',
        save_freq= 'epoch', 
        mode='min',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    ra_gan.fit(train_batch, epochs=args.num_epochs,callbacks=callbacks,verbose=1,steps_per_epoch=steps_per_epoch,validation_steps=validation_steps,validation_data=val_batch)
    checkpoint_paph="{}{}_{}x/{}/{}/model.ckpt".format(args.ckpt_path,args.model,scale_factor,args.generator,'generator') 
    ra_gan.save_weights_gen(checkpoint_paph)
        

    print("Evaluate model")
    eval = ra_gan.evaluate(test_batch, verbose=1)
    saved_model(ra_gan.generator, 'saved_model/{}/'.format(args.model))
    return eval, ra_gan.generator.get_run_time()

if __name__ == '__main__':
    main()
