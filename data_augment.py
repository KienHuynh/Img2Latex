"""
@author: Kien Huynh

This module contains some necessary random augmentation methods which can be used in our math expression recognization training.
"""

import numpy as np
import scipy.misc, scipy.ndimage
from skimage.color import rgb2hsv, hsv2rgb

import pdb, traceback

def gray2rgb(img):
    """gray2rgb
    Convert a gray image to rgb.

    :param img: numpy array, gray image of shape (H, W).
    """
    imh, imw = img.shape
    img = np.reshape(img, (imh, imw, 1))
    return np.concatenate((img, img, img), axis = 2)

def invert_img(img):
    """invert_img
    Invert the input image.

    :param img: numpy array, the image.
    """
    if (np.max(img) > 1.0 or np.min(img) < 0.0):
        print("Warning: invert_img expects an image with values in the [0,1] interval")
    return 1.0 - img


def random_scale(img, min_scale, max_scale, min_pad):
    """random_scale
    This function scales the input image using a random float value ranged from min_scale to max_scale then crops it back to the original size.

    :param img: numpy array, image.
    :min_scale: float, minimum scale.
    :max_scale: float, maximum scale.
    :min_pad: minimum number of pixels to be padded around the equation (after cropping).
    """
    imh, imw, _ = img.shape
    
    # Assign max scaling to a smaller value if the equation will be bigger than imh or imw after scaling
    col_sum = np.sum(img, axis=(0,2))[:]
    row_sum = np.sum(img, axis=(1,2))[:]

    eqh = np.where(row_sum != 0)[0] # Tuple
    eqh = eqh[-1] - eqh[0]
    eqw = np.where(col_sum != 0)[0]
    eqw = eqw[-1] - eqw[0]
    
    max_scale = min(float(imh) / float(eqh), float(imw) / float(eqw), max_scale)
    # Start scaling
    scale_factor = np.random.uniform(min_scale, max_scale)

    print(scale_factor)
    #img_scale = rescale(img, scale_factor)
    img_scale = scipy.misc.imresize(img, scale_factor, interp='bicubic')

    # Cropping
    # Find pixels that are != 0 to avoid cropping them from the image
    #pdb.set_trace()

    col_sum = np.sum(img_scale, axis=(0,2))
    col_sum = np.where(col_sum != 0)[0]
    row_sum = np.sum(img_scale, axis=(1,2))
    row_sum = np.where(row_sum != 0)[0]

    y_min = row_sum[0]
    y_max = row_sum[-1]
    x_min = col_sum[0]
    x_max = col_sum[-1]
    
    if (y_max - y_min) + 2 * min_pad > imh or (x_max - x_min) + 2 * min_pad > imw:
        return img

    y_min = row_sum[0] - min_pad
    y_max = row_sum[-1] + min_pad
    x_min = col_sum[0] - min_pad
    x_max = col_sum[-1] + min_pad

    y_min = max(y_min, 0)
    x_min = max(x_min, 0)
    y_max = min(y_max, imh-1)
    x_max = min(x_max, imw-1)
    
    img_scale = img_scale[y_min:y_max, x_min:x_max, :]
    
    # Calculate padding size
    pad_size_up = int((imh - img_scale.shape[0])/2)
    pad_size_down = (imh - img_scale.shape[0]) - pad_size_up
    pad_size_left = int((imw - img_scale.shape[1])/2)
    pad_size_right = (imw - img_scale.shape[1]) - pad_size_left

    img_scale = np.pad(img_scale, ((pad_size_up, pad_size_down),
        (pad_size_left, pad_size_right),
        (0, 0)), 'constant', constant_values=((0,0),(0,0),(0,0)))

    return img_scale


def random_hue(img):
    """random_hue
    Change the hsv triplet of the symbols in the image.

    :param img: numpy array, image.
    """
    if (np.max(img) > 1.0 or np.min(img) < 0.0):
        print("Warning: random_hue expects an image with values in the [0,1] interval")

    num_zero = np.sum(img == 0)
    num_one = np.sum(img == 1)
    
    # Decide background value
    bg_value = 1
    if (num_zero > num_one):
        bg_value = 0

    bg_pixel_ind = img == bg_value
   
    img_rgb = (img * 255).astype(np.uint8)
    img_hsv = rgb2hsv(img_rgb)
    hue = np.random.uniform(0, 1)
    sat = np.random.uniform(0, 1)
    
    # Black background => symbol pixels must be bright
    # White background => symbol pixels must be dark
    val = np.random.uniform(0, 0.7)
    if (bg_value == 0):
        val = np.random.uniform(0.3, 1)

    img_hsv[:,:,0] = hue
    img_hsv[:,:,1] = sat
    img_hsv[:,:,2] = val
    img_hsv[img_hsv > 1] = 1
    img_hsv[img_hsv < 0] = 0
    img_rgb = hsv2rgb(img_hsv)
    img_rgb[bg_pixel_ind] = bg_value

    #pdb.set_trace()
    return img_rgb


def random_rotate(img, angle_std):
    """random_rotate
    Rotate the image and resize it to the original size (there will be some skewing side-effect).
    
    :param img: numpy array, image.
    :param angle_std: float. The actual angle will be drawn from a normal distribution with zero mean and the specified standard deviation. Note that the angle here is in degrees.
    """
    imh, imw, _ = img.shape
    angle = angle_std*np.random.randn(1)
    img = scipy.ndimage.interpolation.rotate(img, angle)
    return scipy.misc.imresize(img, [imh, imw])
