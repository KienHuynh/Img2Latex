import get_gt
import CROHME_parser
import data_augment
import config as cfg

import numpy as np

import pickle

import pdb
import torch
from torch.autograd import Variable
import time

def batch_data(file_list, scale_list, istrain):
    """batch_data
    This function will batch some images for training/testing purpose.

    :param file_list: list of strings, each string is the full path to an inkml file.
    :param scale_list: list of float values, scale factors to be used when render inkml data into a numpy image.
    :param istrain: boolean
    """
    imh = cfg.IMH
    imw = cfg.IMW
    batch_size = len(file_list)
    if (cfg.USE_COORD):
        batch = np.zeros((imh,imw,5,batch_size), dtype=np.float32)
        grid_x, grid_y = np.meshgrid(range(imw),range(imh))
        grid_x = 255*(grid_x/(imw - 1)).reshape((imh, imw, 1))
        grid_y = 255*(grid_y/(imh - 1)).reshape((imh, imw, 1))
    else:
        batch = np.zeros((imh,imw,3,batch_size), dtype=np.float32)
    
    for i, f in enumerate(file_list):
        scale_factor = scale_list[i]
        img = CROHME_parser.inkml2img(f, scale_factor, target_width=imw, target_height=imh)[0]            
        img = data_augment.gray2rgb(img)
        
        if (istrain and cfg.RAND_TRANSFORM):
            img, keep_original = data_augment.random_transform(img)

        if (cfg.USE_COORD):
            img = np.concatenate((img, grid_x, grid_y), 2)

        batch[:,:,:,i] = img
	
	# Currently, batch has HWCN format
    # Convert it to NCHW format
    batch = np.transpose(batch, (3, 2, 0, 1))/255.0 
    batch = np_to_var(batch, cfg.CUDA)
    dice = np.random.uniform(0, 1.0)
    if (dice < 0.6 and cfg.RAND_TRANSFORM and istrain and not keep_original):
        sigma = np.random.uniform(0.06, 0.11)
        batch = data_augment.elastic_transform_pt(batch, batch.shape[2]*6, batch.shape[2]*sigma)

    return batch 


def batch_target(file_list):
    """batch_target
    This funciton batch target vectors for training/testing purpose.

    :param file_list: list of strings, each string is the full path to an inkml file.
    """
    batch = []
    for f in file_list:
        batch += get_gt.read_latex_label(f, 'mathsymbolclass.txt', cfg.MAX_TOKEN_LEN-1)
     
    return np.asarray(batch)


def get_layers(net, g):
    """get_layers
    Return a list of NN layers statisfying the condition specified in the lambda function.

    :param net: the network.
    :param g: lambda function that takes in a torch NN layer and return a boolean value.
    """
    return [module[1] for module in net.named_modules() if g(module[1])]


def np_to_var(np_array, use_cuda):
    """np_to_var
    This function convert a numpy array to a torch tensor Variable.
       
    :param np_array: the numpy array to be converted.
    :param use_cuda: boolean, indicating if CUDA will be used.
    """
    if (use_cuda):
        return Variable(torch.from_numpy(np_array).cuda())
    else:
        return Variable(torch.from_numpy(np_array))


def var_to_np(torch_var, use_cuda):
    """var_to_np
    This function convert a torch tensor Variable to a numpy array.
       
    :param np_array: the numpy array to be converted.
    :param use_cuda: boolean, indicating if CUDA will be used.
    """
    if (use_cuda):
        return torch_var.data.cpu().numpy()
    else:
        return torch_var.data.numpy()


def grad_clip(net, max_grad = 0.1):
    """grad_clip
    Clipping gradient vectors of net parameters in all layers. The clipping is done separately on each layer.

    :param net: the network.
    :param max_grad: maximum magnitude allowed.
    """
    params = [p for p in list(net.parameters()) if p.requires_grad==True]
    for p in params:
        p_grad = p.grad 

        if (type(p_grad) == type(None)):
            #pdb.set_trace()
            here = 1 
        else:
            magnitude = torch.sqrt(torch.sum(p_grad**2)) 
            if (magnitude.data[0] > max_grad):
                #pdb.set_trace()
                p_grad.data = (max_grad*p_grad/magnitude.data[0]).data


def load_list(file_name):
    """load_list
    Load a list object to file_name.

    :param file_name: string, file name.
    """
    end_of_file = False
    list_obj = [] 
    f = open(file_name, 'rb')
    while (not end_of_file):
        try:
            list_obj.append(pickle.load(f))
        except EOFError:
            end_of_file = True
            print("EOF Reached")

    f.close()
    return list_obj 

def save_list(list_obj, file_name):
    """save_list
    Save a list object to file_name
    
    :param list_obj: List of objects to be saved.
    :param file_name: file name.
    """

    f = open(file_name, 'wb')
    for obj in list_obj:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close() 


def exact_match(s, t):
    """exact_match
    Compare two lists of predicted and target tokens if they are the same

    :param s: predicted string
    :param t: target string
    """
    end_token_id = t.index('</s>')
    t = t[0:end_token_id]
    if (end_token_id >= len(s)):
        return 0
    if (s[end_token_id] != '</s>'):
        return 0 
    s = s[0:end_token_id] 
    for i, ele in enumerate(s):
        if ele != t[i]:
            return 0

    return 1


def levenshtein_distance(s, t):
    """levenshtein_distance
    Computer levenshtein distance between two strings (or list of symbols)

    :param s: first string
    :param t: second string
    """

    m = len(s)
    n = len(t)
    d = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        d[i, 0] = i

    for j in range(n + 1):
        d[0, j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + substitutionCost)

    return d[m, n] / max(m , n)


def softmax(x, axis=1):
    x = np.copy(x)
    xmax = np.max(x, axis=axis, keepdims=True)
    x -= xmax
    xexp = np.exp(x)
    x = xexp/np.sum(xexp,axis=axis,keepdims=True)
    return x
