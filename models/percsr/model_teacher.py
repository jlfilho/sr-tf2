import tensorflow as tf
from models.percsr.blocks import DRB, Upsample


class Teacher(tf.keras.Model):
   def __init__(self,channels=1,scale_factor=2,file_writer_cm=None,distillation_rate=0.8):
      super(Teacher, self).__init__()
      self.c1 = tf.keras.layers.Conv2D(16, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))

      self.RB1 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)
      self.RB2 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)
      self.RB3 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)
      self.RB4 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)
      self.RB5 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)
      self.RB6 = DRB(filters=16,kernel_size=3,distillation_rate=distillation_rate)

      self.c2 = tf.keras.layers.Conv2D(16, 3,padding='same',strides=(1, 1),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', 
            distribution='truncated_normal', seed=None))
      
      self.c3 = tf.keras.layers.Conv2D(16, 3,padding='same',strides=(1, 1),
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
      #x = tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, 0]], 'SYMMETRIC')
      x1 = self.lrelu(self.c1(inputs))

      rb1 = self.RB1(x1)
      rb2 = self.RB2(rb1)
      rb3 = self.RB3(rb2)
      rb4 = self.RB4(rb3)
      rb5 = self.RB5(rb4)
      rb6 = self.RB6(rb5)


      x2 = tf.keras.layers.concatenate([rb1, rb2, rb3, rb4, rb5, rb6],axis=3)

      x2 = self.lrelu(self.c2(x2))
      x2 = self.lrelu(self.c3(x2))

      x_fused = tf.keras.layers.add([x1, x2])

      return self.upsample(x_fused)
