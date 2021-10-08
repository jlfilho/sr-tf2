import tensorflow as tf
import numpy as np

norm_1 = tf.keras.layers.Rescaling(scale=1./255.)
norm_2 = tf.keras.layers.Rescaling(scale=1./127.5,offset=-1)

def scale_1(imgs):
    return imgs / 255.

def unscale_1(imgs):
    imgs = imgs * 255
    imgs = np.clip(imgs, 0., 255.)
    return imgs #.astype('uint8')

def scale_2(imgs):
    return imgs / 127.5 - 1

def unscale_2(imgs):
    imgs = (imgs + 1.) * 127.5
    imgs = np.clip(imgs, 0., 255.)        
    return imgs #.astype('uint8')