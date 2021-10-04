import tensorflow as tf

def espcn(scale_factor=2):   
    inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')

    net = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
    
    net = tf.keras.layers.Conv2D(64, 5,activation='tanh', padding='valid',strides=(1, 1), name='conv1',
                                kernel_initializer=tf.keras.initializers.HeNormal())(net)
    
    net = tf.keras.layers.Conv2D(32, 3,activation='tanh', padding='valid',strides=(1, 1), name='conv2',
                                kernel_initializer=tf.keras.initializers.HeNormal())(net)
    
    net = tf.keras.layers.Conv2D(scale_factor ** 2, 3,activation='sigmoid', padding='valid',strides=(1, 1), name='conv3',
                                kernel_initializer=tf.keras.initializers.HeNormal())(net)
    outputs = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')(net)
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='espcn')
    return model