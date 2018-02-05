import numpy as np
import skimage

def gray2rgb(img):
    """gray2rgb
    Convert a gray image to rgb

    :param img: numpy array, gray image of shape (H, W)
    """
    imh, imw = img.shape
    img = np.reshape(img, (imh, imw, 1))
    return np.concatenate((img, img, img), axis = 2)


def random_scale(img, min_scale, max_scale):
    """random_scale
    This function scales the input image using a random float value ranged from min_scale to max_scale then crops it back to the original size

    :param img: numpy array, image
    :min_scale: float, minimum scale
    :max_scale: float, maximum scale
    """
    imh, imw, _ = img.shape()
    
    # Assign max scaling to a smaller value if the equation will be bigger than imh or imw after scaling
    row_sum = np.sum(img_scale, axis=0)
    col_sum = np.sum(img_scale, axis=1)
    eqh = np.where(row_sum != 0)
    eqh = eqh[-1] - eqh[0]
    eqw = np.where(col_sum != 0)
    eqw = eqw[-1] - eqw[0]
    max_scale = min(float(imh) / float(eqh), float(imw) / float(eqw), max_scale)

    # Start scaling and cropping
    scale_factor = ((max_scale - min_scale)/2) * np.random.normal() + min_scale
    img_scale = skimage.transform.rescale(img, scale_factor, anti_aliasing=True)
    row_sum = np.sum(img_scale, axis=0)
    col_sum = np.sum(img_scale, axis=1)
    row_sum = np.where(row_sum != 0)
    col_sum = np.where(col_sum != 0)
    y_min = row_sum[0]
    y_max = row_sum[-1]
    x_min = col_sum[0]
    x_max = col_sum[-1]


def random_hue():
    abc = 1
