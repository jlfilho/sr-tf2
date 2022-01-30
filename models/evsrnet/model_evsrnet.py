import tensorflow as tf
from models.evsrnet.block import RB, Upsample
import functools


def make_layer(block, n_layers):
    layers = tf.keras.Sequential()
    for _ in range(n_layers):
        layers.add(block())
    return layers


class EVSRNet(tf.keras.Model): 
    def __init__(self,channels=1,blocks=5,scale_factor=2,filters=8,method=None):
        super(EVSRNet,self).__init__()
        self.method = method
        self.scale_factor = scale_factor
        self.conv_1 = tf.keras.layers.Conv2D(filters, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        basic_block = functools.partial(RB, filters=filters)
        self.recon_trunk = make_layer(basic_block, blocks)
        self.upsample = Upsample(channels=channels,scale_factor=scale_factor)
        self.act = tf.keras.layers.ReLU()
        self.time = []
    
    def get_run_time(self):
        if(len(self.time)>0):
            return sum(self.time)/len(self.time)
        else:
            return -1

    def call(self, inputs):
        x = self.act(self.conv_1(inputs))
        x = self.recon_trunk(x)
        x = self.upsample(x)
        if self.method != None:
            input_resized = tf.image.resize(inputs, [inputs.shape[1]*self.scale_factor,inputs.shape[2]*self.scale_factor],method=self.method)
            x = tf.keras.layers.add([x,input_resized])
        return x



class Neuro(tf.keras.Model):
    def __init__(self, n_c, n_b, scale):
        super(Neuro,self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(n_c, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        basic_block = functools.partial(RB, filters=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        self.conv_h = tf.keras.layers.Conv2D(n_c, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        self.conv_o = tf.keras.layers.Conv2D(scale**2*3, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
        self.act = tf.keras.layers.ReLU()


    def call(self, x, h, o):
        x = tf.keras.layers.concatenate([x, h, o])
        x = self.act(self.conv_1(x))
        x = self.recon_trunk(x)
        x_h = self.act(self.conv_h(x))
        x_o = self.conv_o(x)
        return x_h, x_o

class RRN(tf.keras.Model):

    def __init__(self,channels=1,blocks=5,scale_factor=2,filters=8):
        super(RRN, self).__init__()
        self.neuro = Neuro(channels, blocks, scale_factor)
        self.scale = scale_factor
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor))
        self.downsample = tf.keras.layers.Lambda(lambda x:tf.nn.space_to_depth(x,scale_factor))
        self.n_c = channels
        self.time = []


    def get_run_time(self):
        if(len(self.time)>0):
            return sum(self.time)/len(self.time)
        else:
            return -1

    def call(self, inputs, training): #call(self, x, x_h, x_o, init)
        x, x_h, x_o, init = inputs
        #_,_,T,_,_ = x.shape
        f1 = x[:,:,0,:,:]
        f2 = x[:,:,1,:,:]
        x_input = tf.keras.layers.concatenate([f1, f2], axis=1) # See if axis is right
        if init:
            x_h, x_o = self.neuro(x_input, x_h, x_o)
        else:
            x_o = self.downsample(x_o)
            x_h, x_o = self.neuro(x_input, x_h, x_o)
            
        x_o = self.upsample(x_o)    
        f2_resized = tf.image.resize(f2, [f2.shape[1]*self.scale,f2.shape[1]*self.scale])
        x_o = tf.keras.layers.Add([x_o,f2_resized])
        return x_h, x_o

