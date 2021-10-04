import tensorflow as tf

def g_rtsrgan(scale_factor=2):   
    def block(input,filters,kernel_size,name, bn=True):
        d = tf.keras.layers.Conv2D(filters, kernel_size, padding='valid',strides=(1, 1), name=name,
                                kernel_initializer=tf.keras.initializers.HeNormal())(input)
        if bn:
            d = tf.keras.layers.BatchNormalization()(d)
        d = tf.keras.activations.relu(d)
        return d

    inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')
    net = tf.pad(inputs, [[0, 0], [5, 5], [5, 5], [0, 0]], 'SYMMETRIC')
    
    net = block(net,64, 5,'conv1')
    net = tf.keras.activations.tanh(net)

    net = block(net,32,3,'conv2')
    net = tf.keras.activations.tanh(net)

    net = block(net,32,3,'conv3')
    net = tf.keras.activations.tanh(net)
    
    net = tf.keras.layers.Conv2D(scale_factor ** 2, 3, padding='valid',strides=(1, 1), name='conv4',
                                kernel_initializer=tf.keras.initializers.HeNormal())(net)
    net = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')(net)
    outputs = tf.keras.activations.sigmoid(net)
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='g_rtsrgan')
    return model