import tensorflow as tf

def d_ertsrgan(filters=64,input_shape=(72,72,1)):

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
    x = conv2d_block(x, filters*2, strides=2)
    x = conv2d_block(x, filters*4)
    x = conv2d_block(x, filters*4, strides=2)
    x = conv2d_block(x, filters*8)
    x = conv2d_block(x, filters*8, strides=2)
    x = tf.keras.layers.Dense(filters*16)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=input, outputs=x,name='d_ertsrgan')
    return model

def rad_ertsrgan(discriminator=None,shape_hr=(72,72,1)):
        
    def comput_Ra(x):
        d_output1,d_output2 = x
        real_loss = (d_output1 - tf.reduce_mean(d_output2))
        fake_loss = (d_output2 - tf.reduce_mean(d_output1))
        return tf.math.sigmoid(0.5 * tf.add(real_loss, fake_loss))

    # Input Real and Fake images, Dra(Xr, Xf)        
    imgs_hr = tf.keras.Input(shape=shape_hr)
    imgs_sr = tf.keras.Input(shape=shape_hr)

    # C(Xr)
    real = discriminator(imgs_hr, training=True)
    # C(Xf)
    fake = discriminator(imgs_sr, training=True)

    

    # Relativistic Discriminator
    ra_out = tf.keras.layers.Lambda(comput_Ra, name='Ra_out')([real, fake])

    model = tf.keras.Model(inputs=[imgs_hr, imgs_sr], outputs=ra_out,name='ra_discriminator')
    return model