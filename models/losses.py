import tensorflow as tf
import numpy as np


class VGGLossNoActivation(object):

    def __init__(self, image_shape):
        self.model = self.create_model(image_shape)
        
    def create_model(self,image_shape):
        
        
        vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=image_shape)

        x = tf.keras.layers.Conv2D(512, (3, 3),padding='same',
                                   name='block5_conv4')(vgg19.get_layer('block5_conv3').output)
        
        model = tf.keras.Model(inputs=vgg19.input, outputs=x)
        model.trainable = False
        return model
    
    def preprocess_vgg(self, x):
        if isinstance(x, np.ndarray):
            return tf.keras.applications.vgg19.preprocess_input((x))
        else:            
            return tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg19.preprocess_input((x)))(x)
        
    # computes VGG loss or content loss
    def perceptual_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))),None)
    
    def euclidean_content_loss(self, y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))), axis=None))
    
    def compoundLoss(self, y_true, y_pred,alfa=10e-2,beta=10e0):
        return (alfa * tf.math.reduce_mean(tf.math.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))),None) + beta * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=None)))


class GANLoss(object):
    def __init__(self,perc_loss, cont_loss, adv_loss,lbd,eta,mu):
        self.perc_loss=perc_loss
        self.cont_loss=cont_loss
        self.adv_loss=adv_loss
        self.lbd=lbd
        self.eta=eta
        self.mu=mu   

    def discriminator_loss(self,real_output, fake_output):
        noise = 0.05 * tf.random.uniform(tf.shape(real_output))
        real_loss = self.adv_loss(tf.ones_like(real_output)-noise, real_output)
        fake_loss = self.adv_loss(tf.zeros_like(fake_output)+noise, fake_output)
        total_loss = 0.5 * (real_loss + fake_loss)
        return total_loss

    def generator_loss(self,fake_output,img_hr,img_sr):
        noise = 0.05 * tf.random.uniform(tf.shape(fake_output))
        a_loss = self.adv_loss(tf.ones_like(fake_output)-noise, fake_output) 
        c_loss = self.cont_loss(img_hr,img_sr) 
        img_hr = tf.keras.layers.Concatenate()([img_hr, img_hr, img_hr])
        img_sr = tf.keras.layers.Concatenate()([img_sr, img_sr, img_sr])
        p_loss = self.perc_loss(img_hr,img_sr) 
        total_loss = self.eta * c_loss + self.lbd * a_loss + self.mu * p_loss
        return total_loss, c_loss , a_loss , p_loss
    

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))