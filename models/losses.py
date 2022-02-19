import tensorflow as tf
import numpy as np


class VGGLossNoActivation(object):

    def __init__(self, image_shape,loss_fn):
        self.model = self.create_model(image_shape)
        self.loss_fn = loss_fn
        
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

    def custom_perceptual_loss(self, y_true, y_pred):
        y_true = tf.keras.layers.Concatenate()([y_true, y_true, y_true])
        y_pred = tf.keras.layers.Concatenate()([y_pred, y_pred, y_pred])
        return self.loss_fn(self.model(self.preprocess_vgg(y_true)),self.model(self.preprocess_vgg(y_pred)))
    
    def euclidean_content_loss(self, y_true, y_pred):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))), axis=None))
    
    def compoundLoss(self, y_true, y_pred,alfa=10e-2,beta=10e0):
        return (alfa * tf.math.reduce_mean(tf.math.square(self.model(self.preprocess_vgg(y_true)) - self.model(self.preprocess_vgg(y_pred))),None) + beta * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=None)))


class GANLoss(object):
    def __init__(self, loss_pix, loss_fea, loss_dis, adv_loss, alfa, eta, lbd, mu):
        self.loss_pix=loss_pix
        self.loss_fea=loss_fea
        self.loss_dis=loss_dis
        self.adv_loss=adv_loss

        self.alfa=alfa
        self.eta=eta
        self.lbd=lbd
        self.mu=mu   

    def discriminator_loss(self,real_output, fake_output):
        noise = 0.05 * tf.random.uniform(tf.shape(real_output))
        real_loss = self.adv_loss(tf.ones_like(real_output)-noise, real_output)
        fake_loss = self.adv_loss(tf.zeros_like(fake_output)+noise, fake_output)
        total_loss = 0.5 * (real_loss + fake_loss)
        return total_loss

    def generative_loss(self, real_output, fake_output, img_hr,img_sr, teacher_img_sr):
        loss_dis=self.loss_dis(teacher_img_sr,img_sr)
        loss_adv = self.adv_loss(real_output, fake_output) 
        loss_pix = self.loss_pix(img_hr, img_sr)
        #img_hr = tf.keras.layers.Concatenate()([img_hr, img_hr, img_hr])
        #img_sr = tf.keras.layers.Concatenate()([img_sr, img_sr, img_sr])
        loss_fea = self.loss_fea(img_hr,img_sr)
        total_loss = self.alfa * loss_pix + self.eta * loss_fea + self.lbd * loss_dis 
        return total_loss, self.alfa * loss_pix , self.eta * loss_fea, self.lbd * loss_dis, self.mu*loss_adv
    

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))