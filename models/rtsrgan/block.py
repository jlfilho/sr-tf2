import tensorflow as tf

class RRDB(tf.keras.Model):
    def __init__(self,filters=32,kernel_size=3,name=None):
        super(RRDB, self).__init__()
    
        self.c1 = tf.keras.layers.Conv2D(filters, kernel_size,padding='valid',strides=(1, 1), name=name,kernel_initializer=tf.keras.initializers.he_normal())
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.bn(x)
        return self.act(x)