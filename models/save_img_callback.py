import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow.experimental.numpy as tnp
from models.dataset import Dataset
from models.metrics import psnr, ssim
from models.utils import scale_1 as scale
from models.utils import unscale_1 as unscale
from models.utils import plot_test_images


tnp.experimental_enable_numpy_behavior()

class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model=None,model_name=None,epochs_per_save=1,lr_paths=None,hr_paths=None,log_dir=None,file_writer_cm=None,scale_factor=2):
        super(SaveImageCallback, self).__init__()
        self._model = model
        self.scale_factor = scale_factor
        self.model_name = model_name
        self.lr_paths=lr_paths
        self.hr_paths=hr_paths
        self.logdir = log_dir
        self.epochs_per_save = epochs_per_save
        self.file_writer_cm = file_writer_cm

    def on_epoch_end(self, epoch,logs=None):
        if ((epoch+1) % self.epochs_per_save == 0):
            time_elapsed = plot_test_images(self._model,self.lr_paths,self.hr_paths,
             self.logdir,scale_factor=self.scale_factor,model_name=self.model_name,epoch=epoch+1)
            self._model.time.extend(time_elapsed) 
            # batch = self.data_batch
            # self.plot_test_images(batch, epoch+1)
    
    def predict(self,img):
        img_sr=np.squeeze(self._model.predict(
            np.expand_dims(img, 0),
            batch_size=1),axis=0)
        img_sr = unscale(img_sr)  
        return img_sr
    
    def plot_to_image(self,figure):
        fig = plt.figure()
        plt.imshow(figure,cmap='gray', vmin=0, vmax=255)
        plt.grid(False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image
    
    def summary_img(self,imgs_lr,imgs_hr,imgs_sr,epoch):
        with self.file_writer_cm.as_default():
            tf.summary.image('Low_resolution', 
                self.plot_to_image(imgs_lr),max_outputs=1, step=epoch)
            tf.summary.image('High_resolution',
                self.plot_to_image(imgs_hr),max_outputs=1, step=epoch)
            tf.summary.image('High_resolution_prediction', 
                self.plot_to_image(imgs_sr),max_outputs=1, step=epoch)
        
    def plot_test_images(self, batch, epoch):    
        try:
            imgs_lr_n = tf.keras.layers.Lambda(lambda x: scale(x))(batch[0])
            imgs_lr = [np.array(img) for img in batch[0]]
            imgs_hr = [np.array(img) for img in batch[1]]
            imgs_sr = [self.predict(img) for img in imgs_lr_n]
            count=1
            for img_hr, img_lr, img_sr in zip(imgs_hr, imgs_lr, imgs_sr):
                self.summary_img(img_lr,img_hr,img_sr,epoch)
                hr_shape = (int(img_hr.shape[0]),int(img_hr.shape[1]))                      
                img_bi = tf.image.resize(img_lr, hr_shape, method='bicubic').numpy()
                images = {'Low Resoluiton': [img_lr, img_hr],
                          'Bicubic': [img_bi, img_hr],
                          self.model_name: [img_sr, img_hr], 
                          'Original': [img_hr,img_hr]}        
                fig, axes = plt.subplots(1, 4, figsize=(40, 10))
                for i, (title, img) in enumerate(images.items()):
                    axes[i].imshow(img[0],cmap='gray', vmin=0, vmax=255)
                    print("{} - {} {} {}".format(title, img[0].shape, ("- psnr: "+str(round(psnr(img[0],img[1],255.).numpy(),2)) if (title == self.model_name or title == 'Bicubic' ) else " "),
                    ("- ssim: "+str(round(ssim(img[0],img[1],255.).numpy(),2)) if (title == self.model_name or title == 'Bicubic' ) else " ")))
                    axes[i].set_title("{} - {} {} {}".format(title, img[0].shape, ("- psnr: "+str(round(psnr(img[0],img[1],255.).numpy(),2)) if (title == self.model_name or title == 'Bicubic' ) else " "),
                    ("- ssim: "+str(round(ssim(img[0],img[1],255.).numpy(),2)) if (title == self.model_name or title == 'Bicubic' ) else " ")))
                    axes[i].axis('off')
                # Save directory                    
                savefile = os.path.join(self.logdir, "{}_epoch{}_img{}.jpg".format(self.model_name,epoch,count))
                fig.savefig(savefile)
                plt.close()
                count+=1
            
        except Exception as e:
            print(e)