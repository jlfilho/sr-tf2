import tensorflow as tf

def rtvsrsnt(scale_factor=2):   
    inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')
    
    net = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
    net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv1',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net1 = net
    net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv2',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net2 = net 
    net = tf.keras.layers.add([net1, net2])
    
    net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv3',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
    net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
    net3 = net
    net = tf.keras.layers.concatenate([net1, net2, net3],axis=3)
    
    net = tf.keras.layers.Conv2D(scale_factor ** 2, 3,activation='tanh', 
                            padding='same',strides=(1, 1), name='conv4',
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
                                   mode='fan_in', distribution='truncated_normal', seed=None))(net)
    outputs = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),
                                        name = 'prediction')(net)
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='rtvsrgan')
    return model