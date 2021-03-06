import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

import config as cfg
import util

import sys
import get_gt

import pdb
import scipy.misc

class AGRU(nn.Module):
    def __init__(self):        
        # GRU settings
        self.gru_hidden_size = 256
        self.embed_dimension = 256
        self.Q_height = 256
        self.va_len = 512
        
        super(AGRU, self).__init__()
        self.conv1_1 = nn.Conv2d(5, 32, 3, stride=1, padding=1, bias=False)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.conv1_2_bn = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.conv1_3_bn = nn.BatchNorm2d(32)
        self.conv1_4 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.conv1_4_bn = nn.BatchNorm2d(32)
        self.projection1 = nn.Conv2d(5, 32, 1, stride=1, padding=0, bias=False)        

        self.pool_1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_3_bn = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_4_bn = nn.BatchNorm2d(64)
        self.projection2 = nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=False)  

        self.pool_2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv3_3_bn = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv3_4_bn = nn.BatchNorm2d(128)
        self.projection3 = nn.Conv2d(64, 128, 1, stride=1, padding=0, bias=False)  

        self.pool_3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.conv4_1_bn = nn.BatchNorm2d(256)
        self.conv4_1_drop = nn.Dropout2d(p = 0.025)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv4_2_bn = nn.BatchNorm2d(256)
        self.conv4_2_drop = nn.Dropout2d(p = 0.025)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv4_3_bn = nn.BatchNorm2d(256)
        self.conv4_3_drop = nn.Dropout2d(p = 0.025)
        self.conv4_4 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv4_4_bn = nn.BatchNorm2d(256)
        self.projection4 = nn.Conv2d(128, 256, 1, stride=1, padding=0, bias=False)  

        self.conv4_4_drop = nn.Dropout2d(p = 0.025)
        self.last_conv_nc = list(self.conv4_4.parameters())[0].shape[0]        
         
        self.pool_4 = nn.MaxPool2d(2, stride=2)

        self.leaky_relu = nn.LeakyReLU(0.01)

        # Temp Declaration
        # z : update
        # h : reset
        # r : candidate
        # Expect size: 1 x 128 (1 x self.gru_hidden_size)
        # The hard code "128" down there is the Nc of FCN Result
        
        # Convolutional layers
        self.embeds_temp = nn.Linear(cfg.NUM_OF_TOKEN, self.embed_dimension) 
        self.FC_Wyz = nn.Linear(self.embed_dimension, self.gru_hidden_size)
        self.FC_Uhz = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
        #self.FC_Ccz = nn.Linear(128, self.gru_hidden_size)
        
        self.FC_Wyr = nn.Linear(self.embed_dimension, self.gru_hidden_size)
        self.FC_Uhr = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
        self.FC_Ccr = nn.Linear(self.last_conv_nc, self.gru_hidden_size)
        
        self.FC_Wyh = nn.Linear(self.embed_dimension, self.gru_hidden_size)
        self.FC_Urh = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
        self.FC_Ccz = nn.Linear(self.last_conv_nc, self.gru_hidden_size)
            
        self.FC_Wo = nn.Linear(self.embed_dimension, cfg.NUM_OF_TOKEN) #
        self.FC_Wh = nn.Linear(self.gru_hidden_size, self.embed_dimension) # for (11)
        self.FC_Wc = nn.Linear(self.last_conv_nc, self.embed_dimension) #
        
        # GRU layers
        self.coverage_mlp_h = nn.Linear(self.gru_hidden_size, self.va_len)
        self.coverage_mlp_a = nn.Linear(self.last_conv_nc, self.va_len)
        self.coverage_mlp_beta = nn.Linear(self.Q_height, self.va_len)

        self.Va_fully_connected = nn.Linear(self.va_len, 1)
        self.conv_Q_beta = nn.Conv2d(1, self.Q_height, 3, stride=1, padding=1, bias=False) 
        self.alpha_softmax = torch.nn.Softmax(dim=1)

        self.word_to_id, self.id_to_word = get_gt.build_vocab('./mathsymbolclass.txt')              


    def cnn_forward(self, x):
        # Compute a forward pass on convolutional layers        
        x0 = x
        x = self.leaky_relu(self.conv1_1(x))
        x = self.leaky_relu(self.conv1_2(x))
        x = self.leaky_relu(self.conv1_3(x))
        x = self.leaky_relu(self.conv1_4(x))
        x = x + self.projection1(x0)        

        x = self.pool_1(x)
        
        x0 = x
        x = self.leaky_relu(self.conv2_1(x))
        x = self.leaky_relu(self.conv2_2(x))
        x = self.leaky_relu(self.conv2_3(x))
        x = self.leaky_relu(self.conv2_4(x))
        x = x + self.projection2(x0)               

        x = self.pool_2(x)
         
        x0 = x
        x = self.leaky_relu(self.conv3_1(x))
        x = self.leaky_relu(self.conv3_2(x))
        x = self.leaky_relu(self.conv3_3(x))
        x = self.leaky_relu(self.conv3_4(x))
        x = x + self.projection3(x0)               
 
        x = self.pool_3(x)
          
        x0 = x
        x = self.leaky_relu(self.conv4_1(x))
        #x = self.conv4_1_drop(x)
        x = self.leaky_relu(self.conv4_2(x))
        #x = self.conv4_2_drop(x)
        x = self.leaky_relu(self.conv4_3(x))
        #x = self.conv4_3_drop(x)
        x = self.leaky_relu(self.conv4_4(x))
        #x = self.conv4_4_drop(x)
        x = x + self.projection4(x0) 

        fcn_result = self.pool_4(x)

        return fcn_result


    def forward(self, x, target):         
        x_np = np.transpose(x.data.cpu().numpy()[0,:,:,:], (1,2,0))
        # Check if the model is running on a CUDA device
        use_cuda = next(self.parameters())[1].is_cuda        

        fcn_result = self.cnn_forward(x)
 
        # Shape of FCN output using default settings: (batchsize, 128, 16, 32)
        fcn_output_shape = fcn_result.cpu().data.numpy().shape
        batch_size = fcn_output_shape[0]
        fcn_height = fcn_output_shape[2]
        fcn_width = fcn_output_shape[3]

        num_of_block = fcn_height * fcn_width
        
        # Create the starting vector as input for the GRU (It is the token <s>)
        start_vector = np.zeros((batch_size,1,cfg.NUM_OF_TOKEN))
        start_vector[:,0,self.word_to_id['<s>']] = 1
       
        # Creating necessary torch tensor Variable for later usage.
        # GRU_hidden: hidden vector of the GRU at the start.
        # return_tensor: the prediction of mathematical expression.
        # alpha_mat: initial attention map, average over the FCN output.
        # beta_mat: accumulate attention over multiple time steps.
        # all_alpha_mat: storing alpha matrices over all time steps, used for visualization purpose.
        # target: the label of the math expression. It is used in training only .
        if use_cuda:
            GRU_hidden = Variable(torch.FloatTensor(batch_size, self.gru_hidden_size).cuda(cfg.CUDA_DEVICE).zero_())
            return_tensor = Variable(torch.FloatTensor(start_vector).cuda(cfg.CUDA_DEVICE), requires_grad=True)
            #Init alpha and beta Matrix
            alpha_mat = Variable(torch.FloatTensor(batch_size, fcn_height, fcn_width).cuda(cfg.CUDA_DEVICE).fill_(1.0 / float(num_of_block)), requires_grad=True)
            beta_mat = Variable(torch.FloatTensor(batch_size, fcn_height, fcn_width).cuda(cfg.CUDA_DEVICE).zero_(), requires_grad=True)
            all_alpha_mat = Variable(torch.zeros(batch_size, 1, fcn_height,fcn_width).cuda(cfg.CUDA_DEVICE))
            if (self.training):
                target = np.reshape(target.data.cpu(), (batch_size, cfg.MAX_TOKEN_LEN))

        else:
            GRU_hidden = Variable(torch.FloatTensor(batch_size, self.gru_hidden_size).zero_())
            return_tensor = Variable(torch.FloatTensor(start_vector), requires_grad=True)
            #Init Alpha and Beta Matrix
            alpha_mat = Variable(torch.FloatTensor(batch_size, fcn_height, fcn_width).fill_(1.0 / float(num_of_block)), requires_grad=True)
            beta_mat = Variable(torch.FloatTensor(batch_size, fcn_height, fcn_width).zero_(), requires_grad=True)
            all_alpha_mat = Variable(torch.zeros(batch_size, 1, fcn_height,fcn_width))
            if (self.training):
                target = np.reshape(target.data, (batch_size, cfg.MAX_TOKEN_LEN))        

        fcn_flat = fcn_result.permute(0,2,3,1).contiguous()
        fcn_flat = fcn_flat.view(batch_size * fcn_height * fcn_width, fcn_output_shape[1])

        from_h = self.coverage_mlp_h(GRU_hidden.view(batch_size, self.gru_hidden_size))
        from_h = from_h.view(batch_size, self.va_len, 1, 1)
        from_h = from_h.repeat(1, 1, fcn_height, fcn_width)

        from_a = self.coverage_mlp_a(fcn_flat)
        from_a = from_a.contiguous().view(batch_size, fcn_height, fcn_width, self.va_len)
        from_a = from_a.permute(0,3,1,2) .contiguous()       

        F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
        f_flat = F_.transpose(1,3).contiguous()
        f_flat = F_.permute(0,2,3,1).contiguous()
        f_flat = f_flat.view(batch_size * fcn_height * fcn_width, self.Q_height)

        from_b = self.coverage_mlp_beta(f_flat)
        from_b = from_b.contiguous().view(batch_size, fcn_height, fcn_width, self.va_len)
        from_b = from_b.permute(0,3,1,2).contiguous()
                
        from_a = from_a + from_b + from_h

        alpha_mat = F.tanh(from_a)
        alpha_straight = alpha_mat.view(batch_size * fcn_height * fcn_width, self.va_len)
        alpha_mat = self.Va_fully_connected(alpha_straight)
        
        alpha_mat = alpha_mat.transpose(0,1).contiguous().view(batch_size, fcn_height, fcn_width) 
        alpha_mat = self.alpha_softmax(alpha_mat.view(batch_size, 512)).view(batch_size, fcn_height, fcn_width)
        all_alpha_mat = torch.cat([all_alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
            
        ################### START GRU ########################
 
        # Get last predicted symbol: This will be used for GRU's input
        return_vector = torch.squeeze(return_tensor, dim = 1)
        
        ####################################################################
        ################ GRU ITERATION #####################################
        ####################################################################
#        self.t_alpha_mat = []
        self.print_alpha_mat =[]
        
        # Reshape target vector so that it has the shape (batch_size, MAX_TOKEN_LEN)
                    
        for RNN_iterate in range (cfg.MAX_TOKEN_LEN - 1):       
            # Clone of fcn_result: We will use this for generating Ct Vector || Deprecated - We use another approach now!
            multiplied_mat = fcn_result.clone()
        
            expanded_alpha_mat = alpha_mat.view(batch_size, 1, fcn_height, fcn_width)
            expanded_alpha_mat = expanded_alpha_mat.repeat(1, fcn_output_shape[1], 1, 1)
            multiplied_mat = multiplied_mat * expanded_alpha_mat
                           
            # Sum all vector after element-wise multiply to get Ct
            multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 2)
            multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 3)
            
            multiplied_mat = multiplied_mat.view(batch_size, self.last_conv_nc)
           
            ########################################################################################
            ################### GRU SECTION ########################################################
            ########################################################################################

            #--------------------------------------------------------------------
             
            if self.training:
                # While training, we will feed the ground truth into the network 
                last_expected_id = target[:, RNN_iterate]
                last_expected_np = np.zeros((batch_size, cfg.NUM_OF_TOKEN))
                for i in range(batch_size):
                    last_expected_np[i, last_expected_id[i]] = 1
                
                if use_cuda:
                    return_vector = Variable(torch.FloatTensor(last_expected_np).cuda(cfg.CUDA_DEVICE))
                else:
                    return_vector = Variable(torch.FloatTensor(last_expected_np))

            else:
                assert True, 'The forward function here does not support testing mode, use beam_search instead' 
            #######################
            # 
            # y(t-1) = GRU_output
            # h(t-1) = GRU_hidden
            # Ct     = multiplied_mat
            embedded = self.embeds_temp(return_vector)
            zt = self.FC_Wyz(embedded) + self.FC_Uhz(GRU_hidden) + self.FC_Ccz(multiplied_mat) # equation (4) in paper
            zt = F.sigmoid(zt)
            
            rt = self.FC_Wyr(embedded) + self.FC_Uhr(GRU_hidden) + self.FC_Ccr(multiplied_mat) # (5)
            rt = F.sigmoid(rt)
            
            ht_candidate = self.FC_Wyh(embedded) + self.FC_Urh(rt * GRU_hidden) + self.FC_Ccz(multiplied_mat) #6
            ht_candidate = F.tanh(ht_candidate)
            
            GRU_hidden = (1 - zt) * GRU_hidden + zt * ht_candidate # (7)
            
            GRU_output = self.FC_Wo(embedded + self.FC_Wh(GRU_hidden) + self.FC_Wc(multiplied_mat)) 
            
            #GRU_output = F.softmax(GRU_output)
        
            ########################################################################################
            ################### GRU SECTION ########################################################
            ########################################################################################

            #return_vector = Variable(torch.squeeze(GRU_output.data, dim = 1))
            return_vector = GRU_output.view(batch_size, cfg.NUM_OF_TOKEN)
            
            # return_vector = F.softmax(Variable(torch.squeeze(GRU_output.data, dim = 1)))
            # return_tensor = torch.cat([return_tensor, torch.unsqueeze(F.softmax(Variable(torch.squeeze(GRU_output.data, dim = 1))), dim = 1)], 1)
            return_tensor = torch.cat([return_tensor, torch.unsqueeze(return_vector, dim = 1)], 1)

            beta_mat = beta_mat + alpha_mat
            #print (return_tensor.cpu().data.numpy().shape)
            #return_tensor.data[:, insert_index, :] = return_vector.data
            #insert_index = insert_index + 1
            
            #print ('-----')
            #print (return_vector.cpu().data.numpy().shape)
            #print (return_tensor.cpu().data.numpy().shape)
            
            #ret_temp = multiplied_mat.view(1, 65536)

            ##########################################################
            ######### COVERAGE #######################################
            ##########################################################
            # This is a MLP Which receive 3 input:
            # Beta matrix: << Still have no idea how to implement this though :'(
            # FCN result
            # h(t-1): current hidden state of GRU


            # Get Input from h(t - 1)    
            from_h = self.coverage_mlp_h(GRU_hidden.view(batch_size, self.gru_hidden_size))
            from_h = from_h.view(batch_size, self.va_len, 1, 1)
            from_h = from_h.repeat(1, 1, fcn_height, fcn_width)
            #from_h = self.coverage_mlp_h(torch.squeeze(GRU_hidden, dim = 1))

            # New Approach
            fcn_flat = fcn_result.permute(0,2,3,1).contiguous()
            fcn_flat = fcn_flat.view(batch_size * fcn_height * fcn_width, fcn_output_shape[1])
            from_a = self.coverage_mlp_a(fcn_flat)
            from_a = from_a.contiguous().view(batch_size, fcn_height, fcn_width, self.va_len)
            from_a = from_a.permute(0,3,1,2) .contiguous()       
            # --
            F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
            #f_flat = F_.transpose(1,3).contiguous()
            f_flat = F_.permute(0,2,3,1).contiguous()
            f_flat = f_flat.view(batch_size * fcn_height * fcn_width, self.Q_height)
            from_b = self.coverage_mlp_beta(f_flat)
            from_b = from_b.contiguous().view(batch_size, fcn_height, fcn_width, self.va_len)
            from_b = from_b.permute(0,3,1,2).contiguous()
            
            #---------------
            #from_a = from_a - (
                        #    from_a.norm()+
                        #    from_h.norm()*fcn_height * fcn_width * self.va_len
                        #) * torch.unsqueeze(beta_mat, dim=1)  + from_h.repeat(1, fcn_height * fcn_width * self.va_len).view(batch_size, self.va_len, fcn_height, fcn_width)

            #from_a = from_a + from_b + from_h.repeat(1, fcn_height * fcn_width).view(batch_size, self.va_len, fcn_height, fcn_width)
            from_a = from_a + from_b + from_h
            #---------------


            #alpha_mat = torch.squeeze(from_a, dim = 1)
            #print (alpha_mat.cpu().data.numpy())
            
        
            
            alpha_mat = F.tanh(from_a)
            
            ## Va fix ##

            #alpha_straight = alpha_mat.transpose(1,3).contiguous()
            alpha_straight = alpha_mat.view(batch_size * fcn_height * fcn_width, self.va_len)
            alpha_mat = self.Va_fully_connected(alpha_straight)

            alpha_mat = alpha_mat.transpose(0,1).contiguous().view(batch_size, fcn_height, fcn_width)
            
            alpha_mat = self.alpha_softmax(alpha_mat.view(batch_size, 512)).view(batch_size, fcn_height, fcn_width)

            self.print_alpha_mat.append(alpha_mat.cpu().data.numpy())
            all_alpha_mat = torch.cat([all_alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
            
        #return torch.unsqueeze(return_tensor, dim = 1)
        # Returnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ! after a long long way :'(((((

        return return_tensor, all_alpha_mat


    def beam_search(self, x, start_id=20, stop_id=19, beam_size=10):
        beam_size = cfg.BEAM_SIZE
        use_cuda = next(self.parameters())[1].is_cuda        

        fcn_result = self.cnn_forward(x).repeat(beam_size, 1, 1, 1)
        # Shape of FCN output using default settings: (batchsize, 128, 16, 32)
        fcn_output_shape = fcn_result.cpu().data.numpy().shape
        fcn_height = fcn_output_shape[2]
        fcn_width = fcn_output_shape[3]
        
        num_of_block = fcn_height * fcn_width
        
        # Create the starting vector as input for the GRU (It is the token <s>)
        start_vector = np.zeros((beam_size,1,cfg.NUM_OF_TOKEN))
        start_vector[:,0,self.word_to_id['<s>']] = 1
       
        # Creating necessary torch tensor Variable for later usage.
        # GRU_hidden: hidden vector of the GRU at the start.
        # return_tensor: the prediction of mathematical expression.
        # alpha_mat: initial attention map, average over the FCN output.
        # beta_mat: accumulate attention over multiple time steps.
        # all_alpha_mat: storing alpha matrices over all time steps, used for visualization purpose.
        # target: the label of the math expression. It is used in training only .
        if use_cuda:
            GRU_hidden = Variable(torch.FloatTensor(beam_size, self.gru_hidden_size).cuda(cfg.CUDA_DEVICE).zero_())
            return_tensor = Variable(torch.FloatTensor(start_vector).cuda(cfg.CUDA_DEVICE), requires_grad=True)
            #Init alpha and beta Matrix
            alpha_mat = Variable(torch.FloatTensor(beam_size, fcn_height, fcn_width).cuda(cfg.CUDA_DEVICE).fill_(1.0 / float(num_of_block)), requires_grad=True)
            beta_mat = Variable(torch.FloatTensor(beam_size, fcn_height, fcn_width).cuda(cfg.CUDA_DEVICE).zero_(), requires_grad=True)
            all_alpha_mat = Variable(torch.zeros(beam_size, 1, fcn_height,fcn_width).cuda(cfg.CUDA_DEVICE))
            if (self.training):
                target = np.reshape(target.data.cpu(), (beam_size, cfg.MAX_TOKEN_LEN))

        else:
            GRU_hidden = Variable(torch.FloatTensor(beam_size, self.gru_hidden_size).zero_())
            return_tensor = Variable(torch.FloatTensor(start_vector), requires_grad=True)
            #Init Alpha and Beta Matrix
            alpha_mat = Variable(torch.FloatTensor(beam_size, fcn_height, fcn_width).fill_(1.0 / float(num_of_block)), requires_grad=True)
            beta_mat = Variable(torch.FloatTensor(beam_size, fcn_height, fcn_width).zero_(), requires_grad=True)
            all_alpha_mat = Variable(torch.zeros(beam_size, 1, fcn_height,fcn_width))
            if (self.training):
                target = np.reshape(target.data, (beam_size, cfg.MAX_TOKEN_LEN))        

        fcn_flat = fcn_result.permute(0,2,3,1).contiguous()
        fcn_flat = fcn_flat.view(beam_size * fcn_height * fcn_width, fcn_output_shape[1])

        from_h = self.coverage_mlp_h(GRU_hidden.view(beam_size, self.gru_hidden_size))
        from_h = from_h.view(beam_size, self.va_len, 1, 1)
        from_h = from_h.repeat(1, 1, fcn_height, fcn_width)

        from_a = self.coverage_mlp_a(fcn_flat)
        from_a = from_a.contiguous().view(beam_size, fcn_height, fcn_width, self.va_len)
        from_a = from_a.permute(0,3,1,2) .contiguous()       

        F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
        f_flat = F_.transpose(1,3).contiguous()
        f_flat = F_.permute(0,2,3,1).contiguous()
        f_flat = f_flat.view(beam_size * fcn_height * fcn_width, self.Q_height)

        from_b = self.coverage_mlp_beta(f_flat)
        from_b = from_b.contiguous().view(beam_size, fcn_height, fcn_width, self.va_len)
        from_b = from_b.permute(0,3,1,2).contiguous()

        from_a = from_a + from_b + from_h

        alpha_mat = F.tanh(from_a)
        alpha_straight = alpha_mat.view(beam_size * fcn_height * fcn_width, self.va_len)
        alpha_mat = self.Va_fully_connected(alpha_straight)

        alpha_mat = alpha_mat.transpose(0,1).contiguous().view(beam_size, fcn_height, fcn_width) 
        alpha_mat = self.alpha_softmax(alpha_mat.view(beam_size, 512)).view(beam_size, fcn_height, fcn_width)
        all_alpha_mat = torch.cat([all_alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
            
        ################### START GRU ########################
 
        # Get last predicted symbol: This will be used for GRU's input
        return_vector = torch.squeeze(return_tensor, dim = 1)
        
        ####################################################################
        ################ GRU ITERATION #####################################
        ####################################################################
#        self.t_alpha_mat = []
        self.print_alpha_mat =[]
        ended_searches = np.ones((cfg.BEAM_SIZE, 1)) 
        # Reshape target vector so that it has the shape (beam_size, MAX_TOKEN_LEN)
                    
        for RNN_iterate in range (cfg.MAX_TOKEN_LEN - 1):       
                       
            ########################################################################################
            ################### GRU SECTION ########################################################
            ########################################################################################

            #--------------------------------------------------------------------

            if self.training:
                # While training, we will feed the ground truth into the network 
                assert True, "This function is for testing only"
            else:
                return_vector_np = util.var_to_np(return_vector, use_cuda)
                if (RNN_iterate == 0):
                    score_index = np.ones((cfg.BEAM_SIZE, ), dtype=np.int)*start_id
                    score_vector = np.zeros((cfg.BEAM_SIZE, ))
                elif (RNN_iterate == 1):
                    score_index = np.argsort(return_vector_np, 1)[:, ::-1][:, 0:cfg.BEAM_SIZE]
                    score_vector = np.sort(util.softmax(return_vector_np), 1)[:, ::-1][:, 0:cfg.BEAM_SIZE]
                    score_index = score_index[0,:]
                    score_vector = score_vector[0,:]
                # This is where the branching starts
                else:
                    #pdb.set_trace()
                    score_vector = score_vector.reshape(cfg.BEAM_SIZE, 1) + util.softmax((return_vector_np))*ended_searches

                    # Get BEAM_SIZE best prediction out of cfg.BEAM_SIZE * cfg.NUM_OK_TOKEN
                    score_vector_rs = score_vector.flatten()
                    score_index = np.argsort(score_vector_rs)[::-1][0:cfg.BEAM_SIZE]
                    # Convert ind back to sub
                    rows = (score_index/cfg.NUM_OF_TOKEN).astype(int)
                    score_index = score_index % cfg.NUM_OF_TOKEN
                    
                    multiplied_mat = multiplied_mat[rows,:]
                    beta_mat = beta_mat[rows,:]
                    alpha_mat = alpha_mat[rows,:]
                    GRU_hidden = GRU_hidden[rows,:]
                    score_vector = score_vector[rows, score_index]

                last_predicted_id = return_vector.max(1)[1].data 
                last_expected_np = np.zeros((beam_size, cfg.NUM_OF_TOKEN))
                        
                for i in range(beam_size):
                    last_expected_np[i, score_index[i]] = 1
                
                if use_cuda:
                    return_vector = Variable(torch.FloatTensor(last_expected_np).cuda(cfg.CUDA_DEVICE))
                else:
                    return_vector = Variable(torch.FloatTensor(last_expected_np))
            
            multiplied_mat = fcn_result.clone()
            expanded_alpha_mat = alpha_mat.view(beam_size, 1, fcn_height, fcn_width)
            expanded_alpha_mat = expanded_alpha_mat.repeat(1, fcn_output_shape[1], 1, 1)
            multiplied_mat = multiplied_mat * expanded_alpha_mat
                           
            # Sum all vector after element-wise multiply to get Ct
            multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 2)
            multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 3)
            
            multiplied_mat = multiplied_mat.view(beam_size, 256)

            #######################
            # 
            # y(t-1) = GRU_output
            # h(t-1) = GRU_hidden
            # Ct     = multiplied_mat
            embedded = self.embeds_temp(return_vector)
            zt = self.FC_Wyz(embedded) + self.FC_Uhz(GRU_hidden) + self.FC_Ccz(multiplied_mat) # equation (4) in paper
            zt = F.sigmoid(zt)
            
            rt = self.FC_Wyr(embedded) + self.FC_Uhr(GRU_hidden) + self.FC_Ccr(multiplied_mat) # (5)
            rt = F.sigmoid(rt)
            
            ht_candidate = self.FC_Wyh(embedded) + self.FC_Urh(rt * GRU_hidden) + self.FC_Ccz(multiplied_mat) #6
            ht_candidate = F.tanh(ht_candidate)
            
            GRU_hidden = (1 - zt) * GRU_hidden + zt * ht_candidate # (7)
            
            GRU_output = self.FC_Wo(embedded + self.FC_Wh(GRU_hidden) + self.FC_Wc(multiplied_mat)) 
            
            return_vector = GRU_output.view(beam_size, cfg.NUM_OF_TOKEN) 
            return_tensor = torch.cat([return_tensor, torch.unsqueeze(return_vector, dim = 1)], 1)
            
            beta_mat = beta_mat + alpha_mat
            #print (return_tensor.cpu().data.numpy().shape)
            #return_tensor.data[:, insert_index, :] = return_vector.data
            #insert_index = insert_index + 1
            
            #print ('-----')
            #print (return_vector.cpu().data.numpy().shape)
            #print (return_tensor.cpu().data.numpy().shape)
            
            #ret_temp = multiplied_mat.view(1, 65536)

            ##########################################################
            ######### COVERAGE #######################################
            ##########################################################
            # This is a MLP Which receive 3 input:
            # Beta matrix: << Still have no idea how to implement this though :'(
            # FCN result
            # h(t-1): current hidden state of GRU


            # Get Input from h(t - 1)    
            from_h = self.coverage_mlp_h(GRU_hidden.view(beam_size, self.gru_hidden_size))
            from_h = from_h.view(beam_size, self.va_len, 1, 1)
            from_h = from_h.repeat(1, 1, fcn_height, fcn_width)
            #from_h = self.coverage_mlp_h(torch.squeeze(GRU_hidden, dim = 1))

            # New Approach
            fcn_flat = fcn_result.permute(0,2,3,1).contiguous()
            fcn_flat = fcn_flat.view(beam_size * fcn_height * fcn_width, fcn_output_shape[1])
            from_a = self.coverage_mlp_a(fcn_flat)
            from_a = from_a.contiguous().view(beam_size, fcn_height, fcn_width, self.va_len)
            from_a = from_a.permute(0,3,1,2) .contiguous()       
            # --
            F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
            #f_flat = F_.transpose(1,3).contiguous()
            f_flat = F_.permute(0,2,3,1).contiguous()
            f_flat = f_flat.view(beam_size * fcn_height * fcn_width, self.Q_height)
            from_b = self.coverage_mlp_beta(f_flat)
            from_b = from_b.contiguous().view(beam_size, fcn_height, fcn_width, self.va_len)
            from_b = from_b.permute(0,3,1,2).contiguous()
            
            #---------------
            #from_a = from_a - (
                        #    from_a.norm()+
                        #    from_h.norm()*fcn_height * fcn_width * self.va_len
                        #) * torch.unsqueeze(beta_mat, dim=1)  + from_h.repeat(1, fcn_height * fcn_width * self.va_len).view(beam_size, self.va_len, fcn_height, fcn_width)

            #from_a = from_a + from_b + from_h.repeat(1, fcn_height * fcn_width).view(beam_size, self.va_len, fcn_height, fcn_width)
            from_a = from_a + from_b + from_h
            #---------------


            #alpha_mat = torch.squeeze(from_a, dim = 1)
            #print (alpha_mat.cpu().data.numpy())
            
        
            
            alpha_mat = F.tanh(from_a)
            
            ## Va fix ##

            #alpha_straight = alpha_mat.transpose(1,3).contiguous()
            alpha_straight = alpha_mat.view(beam_size * fcn_height * fcn_width, self.va_len)
            alpha_mat = self.Va_fully_connected(alpha_straight)

            alpha_mat = alpha_mat.transpose(0,1).contiguous().view(beam_size, fcn_height, fcn_width)
            
            alpha_mat = self.alpha_softmax(alpha_mat.view(beam_size, 512)).view(beam_size, fcn_height, fcn_width)

            self.print_alpha_mat.append(alpha_mat.cpu().data.numpy())
            all_alpha_mat = torch.cat([all_alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
            
            stop_mat = np.argmax(util.var_to_np(return_tensor, use_cuda), axis=2)
            stop_mat = (stop_mat == stop_id)
            ended_searches[:,0] = 1-(stop_mat[:,-1].astype(np.float32))
            stop_mat = (np.sum(stop_mat, axis=1) != 0)
            stop_mat = np.sum(stop_mat)
            
            if (stop_mat == cfg.BEAM_SIZE):
                #pdb.set_trace()
                return return_tensor, all_alpha_mat
            
        #return torch.unsqueeze(return_tensor, dim = 1)
        # Returnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ! after a long long way :'(((((

        return return_tensor, all_alpha_mat
