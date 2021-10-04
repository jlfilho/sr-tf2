import tensorflow as tf

def psnr(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    values = tf.image.psnr(y, y_pred, max_val=max_val)
    return tf.reduce_mean(values)

def ssim(y, y_pred,max_val=1.0):
    y = tf.image.convert_image_dtype(y, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    values = tf.image.ssim(y, y_pred, max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return tf.reduce_mean(values)

rmse = tf.keras.metrics.RootMeanSquaredError(name='rmse')