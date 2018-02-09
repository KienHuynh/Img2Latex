from get_gt import *
from CROHME_parser import *
from data_augment import *
import config as cfg

import torch
from attend_GRU import AGRU
import get_gt

import glob

import pdb

def batch_data(file_list, scale_list, train):
    """batch_data
    This function will batch some images for training/testing purpose.

    :param file_list: list of strings, each string is the full path to an inkml file.
    :param scale_list: list of float values, scale factors to be used when render inkml data into a numpy image.
    """
    imh = cfg.IMH
    imw = cfg.IMW
    batch_size = len(file_list)
    batch = np.zeros((imh,imw,3,batch_size), dtype=np.float32)

    for i, f in enumerate(file_list):
        scale_factor = scale_list[i]
        img = inkml2img(f, scale_factor, target_width=imw, target_height=imh)[0]
        img = gray2rgb(img)
        if (train):
            img = random_transform(img)
        batch[:,:,:,i] = img

    return batch


def get_layers(net, g):
    """get_layers
    Return a list of NN layers statisfying the condition specified in the lambda function

    :param net: the network
    :param g: lambda function that takes in a torch NN layer and return a boolean value
    """
    return [module[1] for module in net.named_modules() if g(module[1])]


def train():    
    # Getting settings from config.py
    max_len = cfg.MAX_TOKEN_LEN
    num_token = cfg.NUM_OF_TOKEN
    imw = cfg.IMW
    imh = cfg.IMH    

    batch_size = cfg.BATCH_SIZE
    lr = cfg.LR
    momentum = cfg.MOMENTUM
    lr_decay = cfg.LR_DECAY
    num_e = cfg.NUM_EPOCH
    last_e = 0

    use_cuda = cfg.CUDA and torch.cuda.is_available()
    save_path = cfg.MODEL_FOLDER
    dataset_path = cfg.DATASET_PATH + 'CROHME2013_data/TrainINKML/'
    subset_list = cfg.SUBSET_LIST
    scale_factors = cfg.SCALE_FACTORS    
    pdb.set_trace()
    # Load the vocab dictionary for display purpose
    _, id_to_word = get_gt.build_vocab('mathsymbolclass.txt')

    # Load network
    net =  AGRU()
    
    # Get a list of convolutional layers
    conv_layers = get_layers(net, lambda x: type(x) == type(net.conv1_3))
 
    # Get conv parameters 
    conv_params = []    
    for c in conv_layers:
        for p in c.parameters():
            if (p.requires_grad):
                conv_params.append(p)

    # Get a list of layers that are not convolutional
    other_layers = get_layers(net, lambda x: type(x) != type(net.conv1_3) and hasattr(x, 'parameters'))
    other_layers = other_layers[1:] # The first layer is attend_GRU.AGRU
    
    # Get GRU parameters
    gru_params= [] 
    for l in other_layers:
        for p in l.parameters():             
            gru_params.append(p)
                           
    # Set a different learning rate for conv layers and GRU layers
    self.optimizer = optim.Adam([
                    {'params': gru_params},
                    {'params': conv_params, 'lr': 0.001}
                    #{'params': conv_params, 'lr': 1e-10}
                                    ], lr=self.learning_rate)
    self.criterion = nn.CrossEntropyLoss(ignore_index=1)


    # Get full paths to train inkml files, create a list of scale factors to be used for rendering train images
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
        scale_list = scale_list[permu_ind]
        num_ite = int(np.ceil(1.0*num_train/batch_size))
        
        for i in range(num_ite):
            batch_idx = range(i*batch_size, (i+1)*batch_size)
            if (batch_idx[-1] >= num_train):
                batch_idx = range(i*batch_size, num_train)
        
            batch_x = batch_data(inkml_list[batch_idx], scale_list[batch_idx], True)
            pdb.set_trace()
        last_e = e
     
    pdb.set_trace() 


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(1311)
    torch.manual_seed(1311)

    train()
