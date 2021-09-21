import tensorflow as tf
from functools import reduce

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn, metrics):
        super(GAN, self).compile(metrics = metrics)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        
    def load_weights_gen(self,checkpoint_filepath):
        self.generator.load_weights(checkpoint_filepath)
    
    def load_weights_dis(self,checkpoint_filepath):
        self.discriminator.load_weights(checkpoint_filepath)
        

    def train_step(self, data):
        if isinstance(data, tuple):
            img_lr, img_hr = data
        
        disciminator_output_shape = list(self.discriminator.output_shape)
        batch_size = tf.shape(img_lr)[0]
        
        disciminator_output_shape = tf.tuple(disciminator_output_shape)
        disciminator_output_shape[0] = batch_size
        

        # Create a high resolution image from the low resolution one
        img_sr = self.generator(img_lr)

        # Combine them with real images
        combined_images = tf.concat([img_sr, img_hr], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones(disciminator_output_shape), tf.zeros(disciminator_output_shape)], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros(disciminator_output_shape)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(img_lr))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.compiled_metrics.update_state(img_hr, img_sr)
        
        return reduce(lambda x,y: dict(x, **y), ({"d_loss": d_loss, "g_loss": g_loss}, {m.name: m.result() for m in self.metrics})) 