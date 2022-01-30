import tensorflow as tf
from functools import reduce

class PercSR(tf.keras.Model):
    def __init__(self, ra_discriminator, generator, teacher):
        super(PercSR, self).__init__()
        self.generator = generator
        self.ra_discriminator = ra_discriminator
        self.teacher = teacher
        
    def compile(self, d_optimizer, g_optimizer, perc_loss, metrics):
        super(PercSR, self).compile(metrics = metrics)
        self.d_optimizer = d_optimizer 
        self.g_optimizer = g_optimizer
        self.perc_loss = perc_loss  
        
    def load_weights_gen(self,checkpoint_filepath):
        self.generator.load_weights(checkpoint_filepath)
    
    def load_weights_dis(self,checkpoint_filepath):
        self.ra_discriminator.load_weights(checkpoint_filepath)
    
    def save_weights_gen(self,checkpoint_filepath):
        # Save the weights
        self.generator.save_weights(checkpoint_filepath)
             
    @tf.function    
    def train_step(self, data):
        if isinstance(data, tuple):
            img_lr, img_hr = data
        
        teacher_img_sr = self.teacher(img_lr, training=False)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            img_sr = self.generator(img_lr, training=True)

            real_output = self.ra_discriminator([img_hr,img_sr], training=True)
            fake_output = self.ra_discriminator([img_sr,img_hr], training=True)

            total_loss, loss_pix , loss_fea, loss_dis, loss_adv = self.perc_loss(real_output, fake_output, img_hr,img_sr,teacher_img_sr)

            
        gradients_of_generator = gen_tape.gradient(total_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(loss_adv, self.ra_discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.ra_discriminator.trainable_variables))
        
        self.compiled_metrics.update_state(img_hr, img_sr)
        
        return reduce(lambda x,y: dict(x, **y), 
                      ({"total_loss": total_loss, "loss_pix": loss_pix, "loss_fea": loss_fea,"loss_dis": loss_dis, "loss_adv": loss_adv},
                       {m.name: m.result() for m in self.metrics})) 
    
    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            img_lr, img_hr = data
 

        # Compute predictions
        img_sr = self.generator(img_lr, training=False)

        # Update the metrics.
        self.compiled_metrics.update_state(img_hr, img_sr)
        
        results = {m.name: m.result() for m in self.metrics}
        return results