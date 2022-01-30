import tensorflow as tf

class IMDB(tf.keras.Model):
    def __init__(self,filters=12,distillation_rate=0.25):
        super(IMDB, self).__init__()
        self.distilled_channels = int(filters * distillation_rate)
        self.remaining_channels = int(filters - self.distilled_channels)
        
        self.c1 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1))
        self.c2 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1))
        self.c3 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1))
        self.c4 = tf.keras.layers.Conv2D(self.distilled_channels, 3,padding='same',strides=(1, 1))
        self.act = tf.keras.layers.LeakyReLU(alpha=0.05)
        self.c5 = tf.keras.layers.Conv2D(filters, 1,padding='same',strides=(1, 1))

    def call(self, inputs):
        out_c1 = self.act(self.c1(inputs))
        distilled_c1, remaining_c1 = tf.split(out_c1, (self.distilled_channels, self.remaining_channels), axis=3)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = tf.split(out_c2, (self.distilled_channels, self.remaining_channels), axis=3)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = tf.split(out_c3, (self.distilled_channels, self.remaining_channels), axis=3)
        out_c4 = self.c4(remaining_c3)
        out = tf.keras.layers.concatenate([distilled_c1, distilled_c2, distilled_c3, out_c4], axis=3)
        out = self.c5(out)
        out_fused = tf.keras.layers.add([out, inputs])
        return out_fused