import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.experimental.numpy as tnp
from dataset import Dataset
from metrics import psnr, ssim

tnp.experimental_enable_numpy_behavior()

class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model=None,model_name=None,
    epochs_per_save=1,log_dir=None,dataset=None):
        super(SaveImageCallback, self).__init__()
        self.data_batch = dataset
        self._model = model
        self.model_name = model_name
        self.logdir = log_dir
        self.epochs_per_save = epochs_per_save

    def on_epoch_end(self, epoch,logs=None):
        if ((epoch+1) % self.epochs_per_save == 0):
            batch = self.data_batch
            self.plot_test_images(batch, epoch+1)
    
    def predict(self,img):
        img_sr=np.squeeze(self._model.predict(
            np.expand_dims(img, 0),
            batch_size=1),axis=0)
        img_sr = np.array(img_sr * 255)  
        return img_sr
        
    def plot_test_images(self, batch, epoch):    
        try:
            imgs_lr_n = tf.keras.layers.Lambda(lambda x: x / 255.0)(batch[0])
            imgs_lr = [np.array(img) for img in batch[0]]
            imgs_hr = [np.array(img) for img in batch[1]]
            imgs_sr = [self.predict(img) for img in imgs_lr_n]
            count=1
            for img_hr, img_lr, img_sr in zip(imgs_hr, imgs_lr, imgs_sr):
                hr_shape = (int(img_hr.shape[0]),int(img_hr.shape[1]))                      
                img_bi = tf.image.resize(img_lr, hr_shape, method='bicubic').numpy()
                images = {'Low Resoluiton': [img_lr, img_hr],
                          'Bicubic': [img_bi, img_hr],
                          self.model_name: [img_sr, img_hr], 
                          'Original': [img_hr,img_hr]}        
                fig, axes = plt.subplots(1, 4, figsize=(40, 10))
                for i, (title, img) in enumerate(images.items()):
                    axes[i].imshow(img[0],cmap='gray', vmin=0, vmax=255)
                    axes[i].set_title("{} - {} {}".format(title, img[0].shape, ("- psnr: "+str(round(psnr(img[0],img[1],255.).numpy(),2)) if (title == self.model_name or title == 'Bicubic' ) else " ")))
                    axes[i].axis('off')

                # Save directory                    
                savefile = os.path.join(self.logdir, "{}_epoch{}_img{}.jpg".format(self.model_name,epoch,count))
                fig.savefig(savefile)
                plt.close()
                count+=1
            
        except Exception as e:
            print(e)