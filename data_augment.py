import numpy as np
from skimage.transform import rescale

import pdb

def gray2rgb(img):
    """gray2rgb
    Convert a gray image to rgb

    :param img: numpy array, gray image of shape (H, W)
    """
    imh, imw = img.shape
    img = np.reshape(img, (imh, imw, 1))
    return np.concatenate((img, img, img), axis = 2)


def random_scale(img, min_scale, max_scale, min_pad=10):
    """random_scale
    This function scales the input image using a random float value ranged from min_scale to max_scale then crops it back to the original size

    :param img: numpy array, image
    :min_scale: float, minimum scale
    :max_scale: float, maximum scale
    :min_pad: minimum number of pixels to be padded around the equation (after cropping)
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
    scale_factor = ((max_scale - min_scale)/2) * np.random.normal() + min_scale
    img_scale = rescale(img, scale_factor)

    # Cropping
    # Find pixels that are != 0 to avoid cropping them from the image
    #pdb.set_trace()

    col_sum = np.sum(img_scale, axis=(0,2))
    col_sum = np.where(col_sum != 0)[0]
    row_sum = np.sum(img_scale, axis=(1,2))
    row_sum = np.where(row_sum != 0)[0]
    
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


def random_hue():
    abc = 1
