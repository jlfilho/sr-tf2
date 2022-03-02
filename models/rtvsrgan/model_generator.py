import tensorflow as tf
from models.rtvsrgan.blocks import RB,DRB, Upsample


class G_RTVSRGAN(tf.keras.Model):
   def __init__(self,channels=1,scale_factor=2,file_writer_cm=None,method=None):
      super(G_RTVSRGAN, self).__init__()
      self.method = method
      self.scale_factor = scale_factor
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
      x = self.upsample(x)
      if self.method != None:
         input_resized = tf.image.resize(inputs, [inputs.shape[1]*self.scale_factor,inputs.shape[2]*self.scale_factor],method=self.method)
         x = tf.keras.layers.add([x,input_resized])
      return x


# class G_RTVSRGAN_2(tf.keras.Model):
#    def __init__(self,channels=1,scale_factor=2,file_writer_cm=None,distillation_rate=0.8):
#       super(G_RTVSRGAN_2, self).__init__()
#       self.RB1 = RB(filters=72,kernel_size=3)
#       self.RB2 = RB(filters=72,kernel_size=3)
#       self.RB3 = RB(filters=72,kernel_size=3)
#       self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
#       self.time = []
    
#    def get_run_time(self):
#       if(len(self.time)>0):
#          return sum(self.time)/len(self.time)
#       else:
#          return -1

#    def call(self, inputs):
#       x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
#       rb1 = self.RB1(x)
#       rb2 = self.RB2(rb1)
#       x = tf.keras.layers.add([rb1, rb2])
#       rb3 = self.RB3(x)
#       x = tf.keras.layers.concatenate([rb1, rb2, rb3],axis=3)
#       return self.upsample(x)


# class G_RTVSRGAN2(tf.keras.Model):
#    def __init__(self,channels=1,scale_factor=2,file_writer_cm=None,distillation_rate=0.8):
#       super(G_RTVSRGAN2, self).__init__()
#       self.c1 = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1),
#             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
#             distribution='truncated_normal', seed=None))

#       self.RB1 = DRB(filters=32,kernel_size=3,distillation_rate=distillation_rate)
#       self.RB2 = DRB(filters=32,kernel_size=3,distillation_rate=distillation_rate)
#       self.RB3 = DRB(filters=32,kernel_size=3,distillation_rate=distillation_rate)
#       self.RB4 = DRB(filters=32,kernel_size=3,distillation_rate=distillation_rate)
#       self.RB5 = DRB(filters=32,kernel_size=3,distillation_rate=distillation_rate)

#       self.c2 = tf.keras.layers.Conv2D(32, 1,padding='same',strides=(1, 1),
#             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
#             distribution='truncated_normal', seed=None))
      
#       self.c3 = tf.keras.layers.Conv2D(32, 3,padding='same',strides=(1, 1),
#             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
#             distribution='truncated_normal', seed=None))
      
#       self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

#       self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
#       self.time = []
    
#    def get_run_time(self):
#       if(len(self.time)>0):
#          return sum(self.time)/len(self.time)
#       else:
#          return -1

#    def call(self, inputs):
#       x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
#       x1 = self.lrelu(self.c1(x))

#       rb1 = self.RB1(x1)
#       rb2 = self.RB2(rb1)
#       rb3 = self.RB3(rb2)
#       rb4 = self.RB4(rb3)
#       rb5 = self.RB4(rb4)
#       x = tf.keras.layers.concatenate([rb1, rb2, rb3, rb4, rb5],axis=3)

#       x = self.lrelu(self.c2(x))
#       x = self.lrelu(self.c3(x))

#       x_fused = tf.keras.layers.add([x, x1])

#       return self.upsample(x_fused)


# class G_RTVSRGAN(tf.keras.Model):
#    def __init__(self,channels=1,scale_factor=2,file_writer_cm=None,distillation_rate=0.8):
#       super(G_RTVSRGAN, self).__init__()
      
#       self.RB1 = RB(filters=64,kernel_size=3)
#       self.RB2 = RB(filters=64,kernel_size=3)
#       self.RB3 = RB(filters=64,kernel_size=3)
#       self.RB4 = RB(filters=64,kernel_size=3)
#       self.RB5 = RB(filters=64,kernel_size=3)
      
#       self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

#       self.upsample = Upsample(channels=channels,scale_factor=scale_factor)   
#       self.time = []
    
#    def get_run_time(self):
#       if(len(self.time)>0):
#          return sum(self.time)/len(self.time)
#       else:
#          return -1

#    def call(self, inputs):
#       x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
      
#       rb1 = self.RB1(x)
#       rb2 = self.RB2(rb1)
#       x = tf.keras.layers.add([rb1, rb2])
#       rb3 = self.RB3(x)
#       x = tf.keras.layers.add([rb2, rb3])
#       rb4 = self.RB4(x)
#       x = tf.keras.layers.add([rb3, rb4])
#       return self.upsample(x)


