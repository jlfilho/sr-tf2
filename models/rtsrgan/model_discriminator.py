import tensorflow as tf

def d_rtsrgan(filters=64,input_shape=(72,72,1)):

    def conv2d_block(input, filters, strides=1, bn=True):
        d = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
        if bn:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        return d
    
    input = tf.keras.layers.Input(shape=input_shape)
    x = conv2d_block(input, filters, bn=False)
    x = conv2d_block(x, filters, strides=2)
    x = conv2d_block(x, filters*2)
    x = tf.keras.layers.Dense(filters*16)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input, outputs=x,name='d_rtsrgan')
    return model