import tensorflow as tf

# def espcn(channels=1,scale_factor=2,file_writer_cm=None):   
#     inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')

#     net = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
    
#     net = tf.keras.layers.Conv2D(64, 5,padding='valid',strides=(1, 1), name='conv1',
#                                 kernel_initializer=tf.keras.initializers.he_normal())(net)
#     net = tf.keras.activations.tanh(net)
    
#     net = tf.keras.layers.Conv2D(32, 3, padding='valid',strides=(1, 1), name='conv2',
#                                 kernel_initializer=tf.keras.initializers.he_normal())(net)
#     net = tf.keras.activations.tanh(net)
#     net = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,padding='valid',strides=(1, 1), name='final',
#                                 kernel_initializer=tf.keras.initializers.he_normal())(net)
#     net = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')(net)
#     outputs = tf.keras.activations.sigmoid(net)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs,name='espcn')
#     #model = Model(inputs=inputs, outputs=outputs,file_writer_cm=file_writer_cm,name='espcn')
#     return model


class ESPCN(tf.keras.Model):

    def __init__(self,channels=1,scale_factor=2,file_writer_cm=None):
        super(ESPCN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 5,padding='valid',strides=(1, 1), name='conv1', activation='tanh', kernel_initializer=tf.keras.initializers.he_normal())
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='valid',strides=(1, 1), name='conv2', activation='tanh', kernel_initializer=tf.keras.initializers.he_normal())
        self.conv3 = tf.keras.layers.Conv2D(channels*(scale_factor ** 2), 3,padding='valid',strides=(1, 1), name='final', kernel_initializer=tf.keras.initializers.he_normal())
        self.upsample = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),name = 'prediction')
        self.time = []
    
    def get_run_time(self):
        if(len(self.time)>0):
            return sum(self.time)/len(self.time)
        else:
            return -1


    
    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return tf.keras.activations.sigmoid(self.upsample(x))