from get_gt import *
from CROHME_parser import *
import config as cfg


def data_batching(file_list, scale_factor):
    """data_batching
    This function will batch some images for training/testing purpose.

    :param file_list: list of strings, each string is the full path to an inkml file.
    :param scale_factor: float, scale factor to be used when render inkml data into a numpy image.
    """
    imh = cfg.IMH
    imw = cfg.IMW
    batch_size = len(file_list)
    batch = np.zeros((imh,imw,3,batch_size), dtype=np.float32)

    for i, f in enumerate(file_list):
        img = inkml2img(f, scale_factor, target_width=imw, target_height=imh)[0]
        img = gray2rgb(img)
        img = random_transform(img)
        batch[:,:,:,i] = img

    return batch

def train():
    use_cuda = cfg.CUDA
    save_path = cfg.MODEL_FOLDER
    
    # Get full paths to train inkml files

