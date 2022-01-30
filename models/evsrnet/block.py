import tensorflow as tf

class RB(tf.keras.Model):
    def __init__(self,filters=64):
        super(RB, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        self.act = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

    def call(self, inputs):
        identity = inputs
        out = self.act(self.conv1(inputs))
        out = self.conv2(out)
        out_fused = tf.keras.layers.add([identity, out])
        return out_fused

class Upsample(tf.keras.Model):
    def __init__(self,channels=1,scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3, 
            padding='same',strides=(1, 1), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
            mode='fan_in', distribution='truncated_normal', seed=None))
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor))

    def call(self, inputs):
        x = self.conv(inputs)
        return self.upsample(x)