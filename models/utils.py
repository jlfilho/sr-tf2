import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os

#from timeit import default_timer as timer
import time
from tensorflow.keras.preprocessing.image import img_to_array
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from tensorflow.keras.preprocessing.image import load_img
from models.metrics import lpips



#from tensorflow.keras.preprocessing.image import array_to_img

norm_1 = tf.keras.layers.Rescaling(scale=1./255.)
norm_2 = tf.keras.layers.Rescaling(scale=1./127.5,offset=-1)

def scale_1(imgs):
    return imgs / 255.

def unscale_1(imgs):
    imgs = imgs * 255
    imgs = np.clip(imgs, 0., 255.)
    return imgs #.astype('uint8')

def scale_2(imgs):
    return imgs / 127.5 - 1

def unscale_2(imgs):
    imgs = (imgs + 1.) * 127.5
    imgs = np.clip(imgs, 0., 255.)        
    return imgs #.astype('uint8')


def arr_to_tr(arr):
  tr = tf.convert_to_tensor(arr, dtype=tf.float32)
  return tr


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0) 
    start = time.time()
    out = model.predict(input)
    end = time.time()
    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.LANCZOS)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.LANCZOS)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    time_elapsed = end - start
    return out_img, time_elapsed 


def plot_results(images, logdir_path, scale_factor=2,model_name=None,epoch=None,index=None,time=None):
    """Plot the result with zoom-in area."""

    # Create a new figure with a default 111 subplot.
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img[0],vmin=0, vmax=255)
        print("{} - {} {} {} {}".format(title, img[0].shape, 
        ("- psnr: "+str(round(img[2][0].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " "),
        ("- ssim: "+str(round(img[2][1].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " "),
        ("- lpips: "+str(round(img[2][2].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " ")))
        axes[i].set_title("{} - {} {} {} {} {}".format(title, img[0].shape, 
        ("- psnr: "+str(round(img[2][0].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " "),
        ("- ssim: "+str(round(img[2][1].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " "),
        ("- lpips: "+str(round(img[2][2].numpy(),2)) if (title == model_name or title == 'Bicubic' ) else " "),
        ("- time: "+str(round(time,4)) if (title == model_name) else " ")))
        axes[i].axis('off')
        # zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(axes[i], 2, loc=2)
        axins.imshow(img[0], origin="lower")

        w,h,_= img[1].shape
        
        if(title=="Low Resoluiton"):
            x1, x2, y1, y2 = (w//3)//scale_factor, ((w//3)+200)//scale_factor, (h//3)//scale_factor, ((h//3)+200)//scale_factor
        else:
            x1, x2, y1, y2 = (w//3), ((w//3)+200), (h//3), ((h//3)+200)
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y2, y1)
    
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        # Make the line.
        mark_inset(axes[i], axins, loc1=1, loc2=3, fc="none", ec="blue")

    plt.yticks(visible=False)
    plt.xticks(visible=False)
    savefile = os.path.join(logdir_path, "{}_epoch{}_img{}.jpg".format(model_name,epoch,index))
    fig.savefig(savefile)
    plt.close()


def plot_test_images(model,lr_img_paths,hr_img_paths, logdir_path=None,scale_factor=2,model_name=None,epoch=None):
    
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    total_bicubic_ssim = 0.0
    total_test_ssim = 0.0
    total_bicubic_lpips = 0.0
    total_test_lpips = 0.0


    index=0
    time_elapsed = []
    for lr_path, hr_path in zip(lr_img_paths,hr_img_paths):
        
        img_lr = load_img(lr_path)
        img_hr = load_img(hr_path)
        w = img_lr.size[0] * scale_factor
        h = img_lr.size[1] * scale_factor
        img_bi = img_lr.resize((w, h))
        img_sr,t_elapsed = upscale_image(model, img_lr)
        lr_img_arr = img_to_array(img_lr)
        bi_img_arr = img_to_array(img_bi)
        hr_img_arr = img_to_array(img_hr)
        sr_img_arr = img_to_array(img_sr)
        time_elapsed.append(t_elapsed)
        
        
        bicubic_psnr = tf.image.psnr(bi_img_arr, hr_img_arr, max_val=255)
        test_psnr = tf.image.psnr(sr_img_arr, hr_img_arr, max_val=255)
        total_bicubic_psnr += bicubic_psnr
        total_test_psnr += test_psnr

        bicubic_ssim = tf.image.ssim(bi_img_arr, hr_img_arr, max_val=255)
        test_ssim = tf.image.ssim(sr_img_arr, hr_img_arr, max_val=255)
        total_bicubic_ssim += bicubic_ssim
        total_test_ssim += test_ssim

        bicubic_lpips = lpips(arr_to_tr(bi_img_arr), arr_to_tr(hr_img_arr))
        test_lpips = lpips(arr_to_tr(sr_img_arr), arr_to_tr(hr_img_arr))
        total_bicubic_lpips += bicubic_lpips
        total_test_lpips += test_lpips

    

        images = {'Low Resoluiton': [lr_img_arr.astype('uint8'), hr_img_arr.astype('uint8'),[0,0]],
                          'Bicubic': [bi_img_arr.astype('uint8'), hr_img_arr.astype('uint8'),[bicubic_psnr,bicubic_ssim,bicubic_lpips]],
                          model_name: [sr_img_arr.astype('uint8'), hr_img_arr.astype('uint8'),[test_psnr,test_ssim,test_lpips]], 
                          'High Resolution': [hr_img_arr.astype('uint8'),hr_img_arr.astype('uint8'),[0,0]]}

        plot_results(images, logdir_path, scale_factor=scale_factor,model_name=model_name,epoch=epoch,index=index,time=t_elapsed)

        
        index+=1
  

    print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / len(hr_img_paths)))
    print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / len(hr_img_paths)))
    print("Avg running sec per frame %.4f" % (sum(time_elapsed) / len(time_elapsed)))

    return time_elapsed