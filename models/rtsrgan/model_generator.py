import tensorflow as tf
from models.rtsrgan.block import RRDB 

# def g_rtsrgan(scale_factor=2):   
#     def block(input,filters,kernel_size,name, bn=True):
#         d = tf.keras.layers.Conv2D(filters, kernel_size, padding='valid',strides=(1, 1), name=name,
#                                 kernel_initializer=tf.keras.initializers.he_normal())(input)
#         if bn:
#             d = tf.keras.layers.BatchNormalization()(d)
#         d = tf.keras.activations.relu(d)
#         return d

#     inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')
#     net = tf.pad(inputs, [[0, 0], [5, 5], [5, 5], [0, 0]], 'SYMMETRIC')
    
#     net = block(net,64, 5,'conv1')
#     net = tf.keras.activations.tanh(net)

#     net = block(net,32,3,'conv2')
#     net = tf.keras.activations.tanh(net)

#     net = block(net,32,3,'conv3')
#     net = tf.keras.activations.tanh(net)
    
#     net = tf.keras.layers.Conv2D(scale_factor ** 2, 3, padding='valid',strides=(1, 1), name='final',
#                                 kernel_initializer=tf.keras.initializers.he_normal())(net)
#     net = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')(net)
#     outputs = tf.keras.activations.tanh(net)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs,name='g_rtsrgan')
#     return model

class G_RTSRGAN(tf.keras.Model):

    def __init__(self,channels=1,scale_factor=2):
        super(G_RTSRGAN, self).__init__()
        self.RRDB1 = RRDB(filters=64,kernel_size=5,name='conv1')
        self.RRDB2 = RRDB(filters=32,kernel_size=3,name='conv2')
        self.RRDB3 = RRDB(filters=32,kernel_size=3,name='conv3')
        

        self.conv = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,padding='valid',strides=(1, 1), name='final', 
            kernel_initializer=tf.keras.initializers.he_normal())
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')
        self.time = []
    
    def get_run_time(self):
        if(len(self.time)>0):
            return sum(self.time)/len(self.time)
        else:
            return -1


    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [5, 5], [5, 5], [0, 0]], 'SYMMETRIC')
        x = tf.keras.activations.tanh(self.RRDB1(x))
        x = tf.keras.activations.tanh(self.RRDB2(x))
        x = tf.keras.activations.tanh(self.RRDB3(x))
        x = self.conv(x)
        return tf.keras.activations.sigmoid(self.upsample(x))