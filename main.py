import get_gt
import CROHME_parser
import data_augment
import config as cfg

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from attend_GRU import AGRU
import util

import os
import glob
import ntpath
import pickle

import pdb
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train():    
    # Getting settings from config.py
    max_len = cfg.MAX_TOKEN_LEN
    num_token = cfg.NUM_OF_TOKEN
    imw = cfg.IMW
    imh = cfg.IMH    
    
    # Training params
    is_train = True
    batch_size = cfg.BATCH_SIZE
    lr = cfg.LR
    momentum = cfg.MOMENTUM
    lr_decay = cfg.LR_DECAY
    max_grad = cfg.MAX_GRAD_CLIP
    num_e = cfg.NUM_EPOCH

    # Tracking/Saving
    last_e = -1
    global_step = 0
    running_loss = 0
    num_ite_to_log = cfg.NUM_ITE_TO_LOG
    num_ite_to_vis = cfg.NUM_ITE_TO_VIS
    num_epoch_to_save = cfg.NUM_EPOCH_TO_SAVE
    all_loss = []
    save_name = cfg.SAVE_NAME
    meta_name = cfg.META_NAME    
    vis_path = cfg.VIS_PATH    

    use_cuda = cfg.CUDA and torch.cuda.is_available()
    save_path = cfg.MODEL_FOLDER
    dataset_path = cfg.DATASET_PATH + 'CROHME2013_data/TrainINKML/'
    subset_list = cfg.SUBSET_LIST
    scale_factors = cfg.SCALE_FACTORS    
    
    # Load the vocab dictionary for display purpose
    _, id_to_word = get_gt.build_vocab('mathsymbolclass.txt')

    # Initialize the network and load its weights
    net =  AGRU()
    save_files = glob.glob(save_path + save_name + '*.dat')
    meta_files = glob.glob(save_path + meta_name + '*.dat')
    if (len(save_files) > 0):
        save_file = sorted(save_files)[-1]
        print('Loading network weights saved at %s...' % save_file)
        loadobj = torch.load(save_file)
        net.load_state_dict(loadobj['state_dict']) 
        last_e, running_loss, all_loss, lr = util.load_list(sorted(meta_files)[-1])
        print('Loading done.')

    if (use_cuda):
        net.cuda()

    # For debugging
    if (not is_train):
        net.train(False)

    # Get a list of convolutional layers
    conv_layers = util.get_layers(net, lambda x: type(x) == type(net.conv1_3))
 
    # Get conv parameters 
    conv_params = []    
    for c in conv_layers:
        for p in c.parameters():
            if (p.requires_grad):
                conv_params.append(p)

    # Get a list of trainable layers that are not convolutional
    other_layers = util.get_layers(net, lambda x: type(x) != type(net.conv1_3) and hasattr(x, 'parameters'))
    other_layers = other_layers[1:] # The first layer is attend_GRU.AGRU
    
    # Get GRU parameters
    gru_params= [] 
    for l in other_layers:
        for p in l.parameters():             
            gru_params.append(p)
                           
    # Set different learning rates for conv layers and GRU layers
    optimizer = optim.Adam([
               {'params': gru_params},
               {'params': conv_params, 'lr': lr} 
                          ], lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    # Get full paths to train inkml files, create a list of scale factors to be used for rendering train images
    inkml_list = []
    scale_list = []
    
    for i, subset in enumerate(subset_list):
        subset_inkml_list = glob.glob(dataset_path + subset + '*.inkml')
        inkml_list += subset_inkml_list 
        scale_list += [scale_factors[i]] * len(subset_inkml_list)
    inkml_list = np.asarray(inkml_list)
    scale_list = np.asarray(scale_list)
    
    #inkml_list = inkml_list[0:120]
    #scale_list = scale_list[0:120]
    num_train = len(inkml_list)
    num_ite = int(np.ceil(1.0*num_train/batch_size))

    # Main train loop
    for e in range(last_e+1, num_e):
        permu_ind = np.random.permutation(range(num_train))
        inkml_list = inkml_list[permu_ind]
        scale_list = scale_list[permu_ind]
               
        if (e % cfg.NUM_EPOCH_TO_DECAY == cfg.NUM_EPOCH_TO_DECAY-1):
            lr = lr*lr_decay
            print('Current learning rate: %.8f' % lr)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr

        for i in range(num_ite):
            optimizer.zero_grad()

            batch_idx = range(i*batch_size, (i+1)*batch_size)
            if (batch_idx[-1] >= num_train):
                batch_idx = range(i*batch_size, num_train)
            batch_size = len(batch_idx) 
            batch_x = util.batch_data(inkml_list[batch_idx], scale_list[batch_idx], is_train)
            batch_x = util.np_to_var(batch_x, use_cuda)
            batch_y_np = util.batch_target(inkml_list[batch_idx])
            batch_y = util.np_to_var(batch_y_np, use_cuda)
            
            pred_y, attention = net(batch_x, batch_y) 
                
            # Convert the 3D tensor to 2D matrix of shape (batch_size*MAX_TOKEN_LEN, NUM_OF_TOKEN) to compute log loss
            pred_y = pred_y.view(-1, num_token)
            # Remove the <start> token from target vector & prediction vvector
            batch_y = batch_y.view(batch_size, max_len)
            batch_y = batch_y[:,1:].contiguous()
            batch_y = batch_y.view(-1)
            pred_y = pred_y.view(batch_size, max_len, num_token)
            pred_y = pred_y[:,1:].contiguous()
            pred_y = pred_y.view(batch_size * (max_len-1), num_token)

            loss = criterion(pred_y, batch_y)
            loss.backward()
            
            util.grad_clip(net, max_grad)
            optimizer.step()
            
            running_loss += loss.data[0]
            all_loss.append(loss.data[0])
            global_step += 1
            
            # Printing stuffs to console 
            if (global_step % num_ite_to_log == (num_ite_to_log-1)):
                print('Finished ite %d/%d, epoch %d/%d, loss: %.5f' % (i, num_ite, e, num_e, running_loss/num_ite_to_log)) 
                running_loss = 0 
                
                # Printing prediction and target
                pred_y_np = util.var_to_np(pred_y, use_cuda)
                pred_y_np = np.reshape(pred_y_np, (batch_size, max_len-1, num_token))
                # Only display the first sample in the batch
                pred_y_np = pred_y_np[0,0:40,:]
                pred_y_np = np.argmax(pred_y_np, axis=1)
                pred_list = [id_to_word[idx] for idx in list(pred_y_np)]
                print('Prediction: %s' % ' '.join(pred_list))

                batch_y_np = np.reshape(batch_y_np, (batch_size, max_len))
                batch_y_np = batch_y_np[0,1:40]
                target_list = [id_to_word[idx] for idx in list(batch_y_np)]
                print('Target: %s\n' % ' '.join(target_list))
                             
            if (global_step % num_ite_to_vis == (num_ite_to_vis-1)):
                tmp_x = util.var_to_np(batch_x, use_cuda)[0,:,:,:]
                tmp_x = np.transpose(tmp_x, (1,2,0))
                attention_np = attention.data.cpu().numpy()[0,2:,:,:] 

                for k in range(10):
                    attention_k = attention_np[k,:,:]/np.max(attention_np[k,:,:])*0.8
                    attention_k = (scipy.misc.imresize(attention_k, 16.0, interp='bicubic'))/255.0
                    tmp_x = scipy.misc.imresize(tmp_x, attention_k.shape)
                    attention_k = np.repeat(np.expand_dims(attention_k, 2), 3, 2)
                    attention_k *= tmp_x
                    #attention_k[attention_k>1] = 1
                    attention_k = (attention_k).astype(np.uint8)
                    scipy.misc.imsave(vis_path + ('%02d.jpg' % k), attention_k)

                plt.clf()
                plt.plot(all_loss)
                plt.show()
                plt.savefig(vis_path + 'loss.png')


        if (e % num_epoch_to_save == (num_epoch_to_save-1)):
            print('Saving at epoch %d/%d' % (e, num_e))
            torch.save({'state_dict': net.state_dict(), 'opt': optimizer.state_dict()}, save_path + save_name + ('_%03d' % e) + '.dat')
            metadata = [e, running_loss, all_loss, lr]
            util.save_list(metadata, save_path + meta_name + ('_%03d' % e) + '.dat')
             
        last_e = e


def test():
    # Getting settings from config.py
    max_len = cfg.MAX_TOKEN_LEN
    num_token = cfg.NUM_OF_TOKEN
    imw = cfg.IMW
    imh = cfg.IMH    
    
    # Training params
    is_train = False
    batch_size = 1

    # Tracking/Saving
    num_ite_to_log = cfg.NUM_ITE_TO_LOG
    num_ite_to_vis = cfg.NUM_ITE_TO_VIS
    save_name = cfg.SAVE_NAME
    test_name = cfg.TEST_NAME 
    vis_path = cfg.VIS_PATH    

    use_cuda = cfg.CUDA and torch.cuda.is_available()
    save_path = cfg.MODEL_FOLDER
    dataset_path = cfg.DATASET_PATH + 'CROHME2013_data/TestINKML/'
    scale_factor = cfg.TEST_SCALE_FACTOR
    
    # Load the vocab dictionary for display purpose
    word_to_id, id_to_word = get_gt.build_vocab('mathsymbolclass.txt')
    start_id = word_to_id['<s>']
    stop_id = word_to_id['</s>']

    # Initialize the network and load its weights
    net =  AGRU()
    save_files = glob.glob(save_path + save_name + '*.dat')
    if (len(save_files) > 0):
        save_file = sorted(save_files)[-1]
        print('Loading network weights saved at %s...' % save_file)
        loadobj = torch.load(save_file)
        net.load_state_dict(loadobj['state_dict']) 
        print('Loading done.')

    if (use_cuda):
        net.cuda()

    # For debugging
    if (not is_train):
        net.train(False)
 
    # Get full paths to test inkml files, create a list of scale factors to be used for rendering test images
    inkml_list = glob.glob(dataset_path + '*.inkml')
    scale_list = [scale_factor] * len(inkml_list)
    inkml_list = np.asarray(inkml_list)
    scale_list = np.asarray(scale_list)
    
    #inkml_list = inkml_list[0:120]
    #scale_list = scale_list[0:120]
    num_test = len(inkml_list)
    num_ite = int(np.ceil(1.0*num_test/batch_size))

    # Exact match and word error rate
    em = []
    wer = []
    all_pred = []
    all_gt = []
    # Main test loop
    for i in range(num_ite):
        batch_idx = range(i*batch_size, (i+1)*batch_size)
        if (batch_idx[-1] >= num_test):
            batch_idx = range(i*batch_size, num_test)
        batch_size = len(batch_idx) 
        batch_x = util.batch_data(inkml_list[batch_idx], scale_list[batch_idx], is_train)
        batch_x = util.np_to_var(batch_x, use_cuda)
        batch_y_np = util.batch_target(inkml_list[batch_idx])
        batch_y = util.np_to_var(batch_y_np, use_cuda)
        
        #pred_y, attention = net(batch_x, batch_y)
        pred_y, attention = net.beam_search(batch_x, start_id, stop_id)
        pred_y = util.var_to_np(pred_y, use_cuda)
        pred_y = np.argmax(pred_y, 2)
        batch_y = np.reshape(batch_y_np, (batch_size, max_len))
        
        print('Finished ite %d/%d.' % (i, num_ite)) 
        j = 0

        pred_string = pred_y[j,:]
        pred_string = [id_to_word[idx] for idx in list(pred_string)]
        gt_string = batch_y[0,:]
        gt_string = [id_to_word[idx] for idx in list(gt_string)]
        all_pred.append(pred_string)
        all_gt.append(gt_string)
        em.append(util.exact_match(pred_string, gt_string))
        if ('</s>' in pred_string):
            pred_string = pred_string[0:pred_string.index('</s>')+1]
        gt_string = gt_string[0:gt_string.index('</s>')+1]
        wer.append(util.levenshtein_distance(pred_string, gt_string))
       
        if (i % 4 == 0):
            continue
 
        # Printing stuffs to console      
        print('Prediction: %s' % ' '.join(pred_string))
        print('Target: %s\n' % ' '.join(gt_string))
                          
        # Save attention to files for visualization
        file_name = ntpath.basename(inkml_list[batch_idx[j]])[:-6]
        vis_path_j = vis_path + file_name + '/'
        if (not os.path.exists(vis_path_j)):
            os.makedirs(vis_path_j)
        
        tmp_x = np.sum(batch_x.data.cpu().numpy()[j,:,:,:], axis=0)
        attention_np = attention.data.cpu().numpy()[j,1:,:,:]
        pred_string = pred_string[1:]
        for k, word in enumerate(pred_string):
            word = word.replace('/', 'slash_')
            attention_k = attention_np[k,:,:]/np.max(attention_np[k,:,:])*0.8
            attention_k = (scipy.misc.imresize(attention_k, 16.0))/255.0
            tmp_x = scipy.misc.imresize(tmp_x, attention_k.shape)
            attention_k += tmp_x
            attention_k[attention_k>1] = 1
            try:
                scipy.misc.imsave(vis_path_j + ('%02d_%s.jpg' % (k, word)), attention_k)
            except FileNotFoundError:
                pdb.set_trace()
            if (word == '<slash_s>'):
                break

        #pdb.set_trace()

    print("Exact match count: %d/%d" % (sum(em), len(em)))
    print("Word error rate: %.5f" % (np.mean(wer)))
    pdb.set_trace()
    util.save_list([em, wer, all_pred, all_gt], save_path + test_name + '.dat')
        
     
    pdb.set_trace() 



if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(1311)
    torch.manual_seed(1311)

    train()
    #test()
