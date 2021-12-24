import tensorflow as tf
from models.ertsrgan.blocks import RB,DRB, Upsample

# def g_ertsrgan(scale_factor=2):   
#     inputs = tf.keras.layers.Input(shape=(None,None,1),name='input')
    
#     net = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
#     net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv1',
#                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
#                                    mode='fan_in', distribution='truncated_normal', seed=None))(net)
#     net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
#     net1 = net

#     net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv2',
#                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
#                                    mode='fan_in', distribution='truncated_normal', seed=None))(net)
#     net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
#     net2 = net 
#     net = tf.keras.layers.add([net1, net2])
    
#     net = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1), name='conv3',
#                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
#                                    mode='fan_in', distribution='truncated_normal', seed=None))(net)
#     net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
#     net3 = net
#     net = tf.keras.layers.concatenate([net1, net2, net3],axis=3)
    
#     net = tf.keras.layers.Conv2D(scale_factor ** 2, 3,activation='tanh', 
#                             padding='same',strides=(1, 1), name='final',
#                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., 
#                                    mode='fan_in', distribution='truncated_normal', seed=None))(net)
#     outputs = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale_factor),
#                                         name = 'prediction')(net)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs,name='g_ertsrgan')
#     return model


class G_ERTSRGAN_(tf.keras.Model):
   def __init__(self,channels=1,scale_factor=2,file_writer_cm=None):
      super(G_ERTSRGAN_, self).__init__()
      self.RB1 = RB(filters=32,kernel_size=3)
      self.RB2 = RB(filters=32,kernel_size=3)
      self.RB3 = RB(filters=32,kernel_size=3)
      self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
      self.time = []
    
   def get_run_time(self):
      if(len(self.time)>0):
         return sum(self.time)/len(self.time)
      else:
         return -1

   def call(self, inputs):
      x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
      rb1 = self.RB1(x)
      rb2 = self.RB2(rb1)
      x = tf.keras.layers.add([rb1, rb2])
      rb3 = self.RB3(x)
      x = tf.keras.layers.concatenate([rb1, rb2, rb3],axis=3)
      return self.upsample(x)


class G_ERTSRGAN2(tf.keras.Model):
   def __init__(self,channels=1,scale_factor=2,file_writer_cm=None):
      super(G_ERTSRGAN2, self).__init__()
      self.c1 = tf.keras.layers.Conv2D(64, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

      self.RB1 = DRB(filters=64,kernel_size=3)
      self.RB2 = DRB(filters=64,kernel_size=3)
      self.RB3 = DRB(filters=64,kernel_size=3)
      self.RB4 = DRB(filters=64,kernel_size=3)

      self.c2 = tf.keras.layers.Conv2D(64, 1,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
      
      self.c3 = tf.keras.layers.Conv2D(64, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
      
      self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

      self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
      self.time = []
    
   def get_run_time(self):
      if(len(self.time)>0):
         return sum(self.time)/len(self.time)
      else:
         return -1

   def call(self, inputs):
      x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
      x1 = self.c1(x)

      rb1 = self.RB1(x1)
      rb2 = self.RB2(rb1)
      rb3 = self.RB3(rb2)
      rb4 = self.RB4(rb3)
      x = tf.keras.layers.concatenate([rb1, rb2, rb3, rb4],axis=3)

      x = self.lrelu(self.c2(x))
      x = self.c3(x)

      x_fused = tf.keras.layers.add([x, x1])

      return self.upsample(x_fused)


class G_ERTSRGAN(tf.keras.Model):
   def __init__(self,channels=1,scale_factor=2,file_writer_cm=None):
      super(G_ERTSRGAN, self).__init__()
      
      self.RB1 = RB(filters=64,kernel_size=3)
      self.RB2 = RB(filters=64,kernel_size=3)
      self.RB3 = RB(filters=64,kernel_size=3)
      self.RB4 = RB(filters=64,kernel_size=3)
      
      self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

      self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
      self.time = []
    
   def get_run_time(self):
      if(len(self.time)>0):
         return sum(self.time)/len(self.time)
      else:
         return -1

   def call(self, inputs):
      x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
      
      rb1 = self.RB1(x)
      rb2 = self.RB2(rb1)
      x = tf.keras.layers.add([rb1, rb2])
      rb3 = self.RB3(x)
      x = tf.keras.layers.add([rb2, rb3])
      rb4 = self.RB4(x)
      x = tf.keras.layers.add([rb3, rb4])
      return self.upsample(x)