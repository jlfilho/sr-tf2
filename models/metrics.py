import tensorflow as tf
from models.utils import scale_2 as scale
from models.utils import unscale_1 as unscale
from lpips_tf import lpips_tf


def psnr(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    if(len(y.shape)==4):
        values = tf.image.psnr(y[:, 4:-4, 4:-4], y_pred[:, 4:-4, 4:-4], max_val=max_val)
    if (len(y.shape)==3):
        values = tf.image.psnr(y[4:-4, 4:-4], y_pred[4:-4, 4:-4], max_val=max_val)
    return tf.reduce_mean(values)

def ssim(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    if(len(y.shape)==4):
        values = tf.image.ssim(y[:, 4:-4, 4:-4], y_pred[:, 4:-4, 4:-4], max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    if (len(y.shape)==3):
        values = tf.image.ssim(y[4:-4, 4:-4], y_pred[4:-4, 4:-4], max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean(values)

rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')

def psnr_loss(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    if(len(y.shape)==4):
        values = tf.image.psnr(y[:, 4:-4, 4:-4], y_pred[:, 4:-4, 4:-4], max_val=max_val)
    if (len(y.shape)==3):
        values = tf.image.psnr(y[4:-4, 4:-4], y_pred[4:-4, 4:-4], max_val=max_val)
    return values

def ssim_loss(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    if(len(y.shape)==4):
        values = tf.image.ssim(y[:, 4:-4, 4:-4], y_pred[:, 4:-4, 4:-4], max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    if (len(y.shape)==3):
        values = tf.image.ssim(y[4:-4, 4:-4], y_pred[4:-4, 4:-4], max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return values


def lpips(y, y_pred):
    y = (y*255.) #/ 127.5 - 1
    y = tf.keras.layers.Concatenate()([y, y, y])
    y_pred = (y_pred*255.) # / 127.5 - 1
    y_pred = tf.keras.layers.Concatenate()([y_pred, y_pred, y_pred])
    if(len(y.shape)==4):
        values = lpips_tf.lpips(y[:, 4:-4, 4:-4], y_pred[:, 4:-4, 4:-4], model='net-lin', net='alex')
    if (len(y.shape)==3):
        values = lpips_tf.lpips(y[4:-4, 4:-4], y_pred[4:-4, 4:-4], model='net-lin', net='alex')
    return tf.reduce_mean(values)




