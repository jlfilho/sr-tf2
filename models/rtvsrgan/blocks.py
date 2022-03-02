import tensorflow as tf

class RB(tf.keras.Model):
    def __init__(self,filters=32,kernel_size=3,name=None):
        super(RB, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1), name=name,
                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.conv(inputs)
        return self.lrelu(x)


class Upsample(tf.keras.Model):
    def __init__(self,channels=1,scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,activation='tanh', 
            padding='same',strides=(1, 1), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
            mode='fan_in', distribution='truncated_normal', seed=None))
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor))

    def call(self, inputs):
        x = self.conv(inputs)
        return self.upsample(x)


class DRB(tf.keras.Model):
    def __init__(self,filters=32,kernel_size=3,distillation_rate=0.8):
        super(DRB, self).__init__()
        self.distilled_channels = int(filters * distillation_rate)
        self.remaining_channels = int(filters - self.distilled_channels)

        self.c1 = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        
        self.c2 = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        
        self.c3 = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        
        self.c4 = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

        self.c5 = tf.keras.layers.Conv2D(filters, kernel_size,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

        self.c6 = tf.keras.layers.Conv2D(filters, 1,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.lrelu(self.c1(inputs))
        distilled_c1, remaining_c1 = tf.split(x, (self.distilled_channels, self.remaining_channels), axis=3)
        x = self.lrelu(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = tf.split(x, (self.distilled_channels, self.remaining_channels), axis=3)
        x = self.lrelu(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = tf.split(x, (self.distilled_channels, self.remaining_channels), axis=3)
        x = self.lrelu(self.c4(remaining_c3))
        distilled_c4, remaining_c4 = tf.split(x, (self.distilled_channels, self.remaining_channels), axis=3)
        x = self.lrelu(self.c5(remaining_c4))
        #print(distilled_c1.shape, distilled_c2.shape, distilled_c3.shape, distilled_c4.shape)
        x = tf.keras.layers.concatenate([distilled_c1, distilled_c2, distilled_c3, distilled_c4],axis=3)
        x = self.lrelu(self.c6(x))
        x_fused = tf.keras.layers.add([x, inputs])
        return self.lrelu(x_fused)


class Upsample(tf.keras.Model):
    def __init__(self,channels=1,scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,activation='tanh', 
            padding='same',strides=(1, 1), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
            mode='fan_in', distribution='truncated_normal', seed=None))
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor))

    def call(self, inputs):
        x = self.conv(inputs)
        return self.upsample(x)