from get_gt import *
from CROHME_parser import *
import config as cfg

import torch

import glob

import pdb

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
    # Getting settings from config.py
    max_len = cfg.MAX_TOKEN_LEN
    num_token = cfg.NUM_OF_TOKEN
    imw = cfg.IMW
    imh = cfg.IMH    

    batch_size = cfg.BATCH_SIZE
    lr = cfg.LR
    num_e = cfg.NUM_EPOCH
    last_e = 0

    use_cuda = cfg.CUDA and torch.cuda.is_available()
    save_path = cfg.MODEL_FOLDER
    dataset_path = cfg.DATASET_PATH + 'CROHME2013_data/TrainINKML/'
    subset_list = cfg.SUBSET_LIST
    scale_factors = cfg.SCALE_FACTORS

    # Get full paths to train inkml files
    inkml_list = []
    scale_list = []
    for i, subset in enumerate(subset_list):
        subset_inkml_list = glob.glob(dataset_path + subset + '*.inkml')
        inkml_list += subset_inkml_list 
        scale_list += [scale_factors[i]] * len(subset_inkml_list)
    inkml_list = np.asarray(inkml_list)
    scale_list = np.asarray(scale_list)

    num_train = len(inkml_list)

    for e in range(last_e, num_e):
        permu_ind = np.random.permutation(range(num_train))
        inkml_list = inkml_list[permu_ind]
        pdb.set_trace()
        last_e = e 
     
    pdb.set_trace() 

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(1311)
    torch.manual_seed(1311)

    train()
