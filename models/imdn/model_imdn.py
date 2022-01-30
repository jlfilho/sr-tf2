import tensorflow as tf
from models.imdn.block import IMDB 

class IMDN(tf.keras.Model):
    '''
    
    '''

    def __init__(self,channels=1,scale_factor=2,filters=12):
        super(IMDN, self).__init__()
        self.fea_conv = tf.keras.layers.Conv2D(12, 3,padding='same',strides=(1, 1))
        # IMDBs
        self.IMDB1 = IMDB(filters=filters)
        self.IMDB2 = IMDB(filters=filters)
        self.IMDB3 = IMDB(filters=filters)
        self.c = tf.keras.layers.Conv2D(12, 1,padding='same',strides=(1, 1))
        self.act = tf.keras.layers.LeakyReLU(alpha=0.05)
        
        self.out_conv = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,strides=(1, 1))
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')

        self.time = []
    
    def get_run_time(self):
        if(len(self.time)>0):
            return sum(self.time)/len(self.time)
        else:
            return -1

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        out_fea = self.fea_conv(x)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_c = self.act(self.c(out_B3))
        out_fused = tf.keras.layers.add([out_c, x])
        out_conv = self.out_conv(out_fused)
        return self.upsample(out_conv)