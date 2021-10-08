import tensorflow as tf
from models.Model import Model

def espcn(scale_factor=2):   
    inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')

    net = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
    
    net = tf.keras.layers.Conv2D(64, 5,padding='valid',strides=(1, 1), name='conv1',
                                kernel_initializer=tf.keras.initializers.he_normal())(net)
    net = tf.keras.activations.relu(net)
    
    net = tf.keras.layers.Conv2D(32, 3, padding='valid',strides=(1, 1), name='conv2',
                                kernel_initializer=tf.keras.initializers.he_normal())(net)
    net = tf.keras.activations.relu(net)
    
    net = tf.keras.layers.Conv2D(scale_factor ** 2, 3,padding='valid',strides=(1, 1), name='final',
                                kernel_initializer=tf.keras.initializers.he_normal())(net)
    net = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')(net)
    outputs = tf.keras.activations.tanh(net)
    model = Model(inputs=inputs, outputs=outputs,name='espcn')
    return model