import tensorflow as tf
from functools import reduce

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        
    def compile(self, d_optimizer, g_optimizer, d_loss, g_loss, metrics):
        super(GAN, self).compile(metrics = metrics)
        self.d_optimizer = d_optimizer
        self.d_loss = d_loss 
        self.g_optimizer = g_optimizer
        self.g_loss = g_loss   
        
    def load_weights_gen(self,checkpoint_filepath):
        self.generator.load_weights(checkpoint_filepath)
    
    def load_weights_dis(self,checkpoint_filepath):
        self.discriminator.load_weights(checkpoint_filepath)
    
    def save_weights_gen(self,checkpoint_filepath):
        # Save the weights
        self.generator.save_weights(checkpoint_filepath)
             
        
    def train_step(self, data):
        if isinstance(data, tuple):
            img_lr, img_hr = data
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            img_sr = self.generator(img_lr, training=True)

            real_output = self.discriminator(img_hr, training=True)
            fake_output = self.discriminator(img_sr, training=True)

            g_loss,c_loss, a_loss, p_loss = self.g_loss(fake_output,img_hr,img_sr)
            d_loss = self.d_loss(real_output, fake_output)
            
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        self.compiled_metrics.update_state(img_hr, img_sr) 
        
        return reduce(lambda x,y: dict(x, **y), 
                      ({"d_loss": d_loss, "g_loss": g_loss,"a_loss": a_loss, "c_loss": c_loss, "p_loss": p_loss },
                       {m.name: m.result() for m in self.metrics})) 