from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy

from torch.nn import Parameter
import NetWorkConfig

import pdb
import math

import sys
sys.path.insert(0, './parser')
import getGT

class WAP(nn.Module):
	def __init__(self):
		self.using_cuda = False

		#######################################################
		## PARAMETER DIFINITION
		self.gru_hidden_size = 256
		self.embed_dimension = 256
		self.Q_height = 256
		self.va_len = 512
		#######################################################
		## NETWORK STRUCTURE
	
		super(WAP, self).__init__()
		self.conv1_1 = nn.Conv2d(9, 32, 3, stride=1,padding=1)
		self.conv1_1_bn = nn.BatchNorm2d(32)
		self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_2_bn = nn.BatchNorm2d(32)
		self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_3_bn = nn.BatchNorm2d(32)
		self.conv1_4 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_4_bn = nn.BatchNorm2d(32)
        
		self.pool_1 = nn.MaxPool2d(2, stride=2)
        
		self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
		self.conv2_1_bn = nn.BatchNorm2d(64)
		self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_2_bn = nn.BatchNorm2d(64)
		self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_3_bn = nn.BatchNorm2d(64)
		self.conv2_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_4_bn = nn.BatchNorm2d(64)
        
		self.pool_2 = nn.MaxPool2d(2, stride=2)
		self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_1_bn = nn.BatchNorm2d(64)
		self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_2_bn = nn.BatchNorm2d(64)
		self.conv3_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_3_bn = nn.BatchNorm2d(64)
		self.conv3_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_4_bn = nn.BatchNorm2d(64)
        
		self.pool_3 = nn.MaxPool2d(2, stride=2)
        
		self.conv4_1 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
		self.conv4_1_bn = nn.BatchNorm2d(128)
		self.conv4_1_drop = nn.Dropout2d(p = 0.025)
		self.conv4_2 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_2_bn = nn.BatchNorm2d(128)
		self.conv4_2_drop = nn.Dropout2d(p = 0.025)
		self.conv4_3 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_3_bn = nn.BatchNorm2d(128)
		self.conv4_3_drop = nn.Dropout2d(p = 0.025)
		self.conv4_4 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_4_bn = nn.BatchNorm2d(128)
		self.conv4_4_drop = nn.Dropout2d(p = 0.025)
        
        
		self.pool_4 = nn.MaxPool2d(2, stride=2)


		

		# Temp Declaration
		# z : update
		# h : reset
		# r : candidate
		# Expect size: 1 x 128 (1 x self.gru_hidden_size)
		# The hard code "128" down there is the height of FCN Result
		
		#self.embeds = nn.Embedding(NetWorkConfig.NUM_OF_TOKEN, self.embed_dimension) 
		self.embeds_temp = nn.Linear(NetWorkConfig.NUM_OF_TOKEN, self.embed_dimension) 
		self.FC_Wyz = nn.Linear(self.embed_dimension, self.gru_hidden_size)
		self.FC_Uhz = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
		self.FC_Ccz = nn.Linear(128, self.gru_hidden_size)
		
		self.FC_Wyr = nn.Linear(self.embed_dimension, self.gru_hidden_size)
		self.FC_Uhr = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
		self.FC_Ccr = nn.Linear(128, self.gru_hidden_size)
		
		self.FC_Wyh = nn.Linear(self.embed_dimension, self.gru_hidden_size)
		self.FC_Urh = nn.Linear(self.gru_hidden_size, self.gru_hidden_size)
		self.FC_Ccz = nn.Linear(128, self.gru_hidden_size)
			
		self.FC_Wo = nn.Linear(self.embed_dimension, NetWorkConfig.NUM_OF_TOKEN) #
		self.FC_Wh = nn.Linear(self.gru_hidden_size, self.embed_dimension) # for (11)
		self.FC_Wc = nn.Linear(128, self.embed_dimension) #
		

		###############################################
		########### GRU ###############################
		###############################################

		
		self.Coverage_MLP_From_H = nn.Linear(self.gru_hidden_size, self.va_len)
		self.Coverage_MLP_From_A = nn.Linear(128, self.va_len)
		self.Coverage_MLP_From_Beta = nn.Linear(self.Q_height, self.va_len)

		self.Va_fully_connected = nn.Linear(self.va_len, 1)
		
		self.conv_Q_beta = nn.Conv2d(1, self.Q_height, 3, stride=1, padding=1) 
		
		
		self.alpha_softmax = torch.nn.Softmax()
		#self.testnn = nn.Linear(65536, 128)
		

		self.word_to_id, self.id_to_word = getGT.buildVocab('./parser/mathsymbolclass.txt')
		
	def setCuda(self, state):
		self.using_cuda = state

	def setGroundTruth(self, GT):
		self.GT = GT
		
	def forward(self, x):
		#print (self.training)

		####################################################################
		################ FCN BLOCK #########################################
		####################################################################
		

		x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
		x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
		x = F.relu(self.conv1_3_bn(self.conv1_3(x)))
		x = F.relu(self.conv1_4_bn(self.conv1_4(x)))
        
		x = self.pool_1(x)
        
		x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
		x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
		x = F.relu(self.conv2_3_bn(self.conv2_3(x)))
		x = F.relu(self.conv2_4_bn(self.conv2_4(x)))
        
		x = self.pool_2(x)
        
		x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
		x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
		x = F.relu(self.conv3_3_bn(self.conv3_3(x)))
		x = F.relu(self.conv3_4_bn(self.conv3_4(x)))
        
		x = self.pool_3(x)
        
		x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
		x = self.conv4_1_drop(x)
		x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
		x = self.conv4_2_drop(x)
		x = F.relu(self.conv4_3_bn(self.conv4_3(x)))
		x = self.conv4_3_drop(x)
		x = F.relu(self.conv4_4_bn(self.conv4_4(x)))
		x = self.conv4_4_drop(x)
		
		FCN_Result = self.pool_4(x)


		# Shape of FCU result: normally: (batchsize, 128, 16, 32)
		current_tensor_shape = FCN_Result.cpu().data.numpy().shape
		num_of_block = current_tensor_shape[2] * current_tensor_shape[3]
		################ DEFINITION ########################################
		
		start_vector = numpy.zeros((current_tensor_shape[0],1,NetWorkConfig.NUM_OF_TOKEN))
		start_vector[:,0,self.word_to_id['<s>']] = 1

		if self.using_cuda:
			#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
			GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], self.gru_hidden_size).cuda().zero_())
			# Init return tensor (the prediction of mathematical Expression)
			return_tensor = Variable(torch.FloatTensor(start_vector).cuda(), requires_grad=True)
			#Init Alpha and Beta Matrix
			alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).cuda().fill_(1.0 / float(num_of_block)), requires_grad=True)
			beta_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).cuda().zero_(), requires_grad=True)
			self.alpha_mat = Variable(torch.zeros(NetWorkConfig.BATCH_SIZE, 1, current_tensor_shape[2],current_tensor_shape[3]).cuda())
		else:
			#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
			GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], self.gru_hidden_size).zero_())
			# Init return tensor (the prediction of mathematical Expression)
			return_tensor = Variable(torch.FloatTensor(start_vector), requires_grad=True)
			#Init Alpha and Beta Matrix
			alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).fill_(1.0 / float(num_of_block)), requires_grad=True)
			beta_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).zero_(), requires_grad=True)
			self.alpha_mat = Variable(torch.zeros(NetWorkConfig.BATCH_SIZE, 1, current_tensor_shape[2],current_tensor_shape[3]))
		FCN_Straight = FCN_Result.permute(0,2,3,1).contiguous()
		FCN_Straight = FCN_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], current_tensor_shape[1])
	
		from_h = self.Coverage_MLP_From_H(GRU_hidden.view(current_tensor_shape[0], self.gru_hidden_size))
		from_h = from_h.view(-1, 1, 1)
		from_h = from_h.repeat(1, 1, current_tensor_shape[2], current_tensor_shape[3])

		from_a = self.Coverage_MLP_From_A(FCN_Straight)
		from_a = from_a.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
		
		F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
		F_Straight = F_.transpose(1,3).contiguous()
		F_Straight = F_.permute(0,2,3,1).contiguous()
		F_Straight = F_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.Q_height)
		from_b = self.Coverage_MLP_From_Beta(F_Straight)
		from_b = from_b.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
                
	        #from_a = from_a + from_b + from_h.repeat(1, current_tensor_shape[2] * current_tensor_shape[3]).view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
		from_a = from_a + from_b + from_h

		alpha_mat = F.tanh(from_a)
		alpha_straight = alpha_mat.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.va_len)
		alpha_mat = self.Va_fully_connected(alpha_straight)
		
		alpha_mat = alpha_mat.transpose(0,1).contiguous().view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
		
		alpha_mat = self.alpha_softmax(alpha_mat.view(current_tensor_shape[0], 512)).view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
		self.alpha_mat = torch.cat([self.alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
			
#			pdb.set_trace()
		
                #pdb.set_trace()
		####################################################################
		################ GRU BLOCK #########################################
		####################################################################

		################### START GRU ########################

		
		# insert_index = 1
		
		# Init the first vector in return_tensor: It is the <s> token
#		return_tensor.data[:, 0, self.word_to_id['<s>']] = 1

		# Get last predicted symbol: This will be used for GRU's input
		return_vector = torch.squeeze(return_tensor, dim = 1)
		#GRU_output = Variable(return_tensor.data[:, 0, :])
		
#		#pdb.set_trace()
		####################################################################
		################ GRU ITERATION #####################################
		####################################################################
#		self.t_alpha_mat = []
		self.print_alpha_mat =[]
		
		for RNN_iterate in range (NetWorkConfig.MAX_TOKEN_LEN - 1):

			#print (alpha_mat.cpu().data.numpy().shape)
		
			# Clone of FCN_Result: We will use this for generating Ct Vector || Deprecated - We use another approach now!
			multiplied_mat = FCN_Result.clone()

			# Element-wise multiply between alpha and FCN_Result
			#--------
			#for batch_index in range(current_tensor_shape[0]):
			#	for i in range (current_tensor_shape[1]):
			#		multiplied_mat[batch_index][i] = multiplied_mat[batch_index][i] * alpha_mat[batch_index]
			#-------- # alpha : batch x 16 x 32
			expanded_alpha_mat = alpha_mat.view(current_tensor_shape[0], 1, current_tensor_shape[2], current_tensor_shape[3])
			expanded_alpha_mat = expanded_alpha_mat.repeat(1, current_tensor_shape[1], 1, 1)
			#pdb.set_trace()
			multiplied_mat = multiplied_mat * expanded_alpha_mat
			#--------
			#mytemp = Variable(alpha_mat.expand(current_tensor_shape).data)
			#expanded_alpha_mat = mytemp
			#multiplied_mat = multiplied_mat * expanded_alpha_mat
					
			# Sum all vector after element-wise multiply to get Ct
			if NetWorkConfig.CURRENT_MACHINE == 0:
				multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 2)
				multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 3)
			else:
				multiplied_mat = torch.sum(multiplied_mat, dim = 2)
				multiplied_mat = torch.sum(multiplied_mat, dim = 3)
			#multiplied_mat = self.testnn(multiplied_mat.view(current_tensor_shape[0], 65536))
			
			multiplied_mat = multiplied_mat.view(current_tensor_shape[0], 128)
			
			
			########################################################################################
			################### GRU SECTION ########################################################
			########################################################################################

			#--------------------------------------------------------------------
			 
			if self.training == True:
				
				last_expected_id = self.GT[:, RNN_iterate]
				last_expected_np = numpy.zeros((current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN))
				for i in range(current_tensor_shape[0]):
					last_expected_np[i, last_expected_id[i]] = 1
				
				if self.using_cuda:
					return_vector = Variable(torch.FloatTensor(last_expected_np).cuda())
				else:
					return_vector = Variable(torch.FloatTensor(last_expected_np))
			else:


				last_predicted_id = return_vector.max(1)[1].data

				last_expected_id = self.GT[:, RNN_iterate]
				
				
				#if last_predicted_id[0] != 19:
				#	print('---------------------')

				last_expected_np = numpy.zeros((current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN))
					

							
				for i in range(current_tensor_shape[0]):
				
					#if last_predicted_id[0] != 19:
					#	print (str(last_expected_id[i]) + ' -- ' + str(last_predicted_id[i]))
#					last_expected_np[i, last_expected_id[i]] = 1
					last_expected_np[i, last_predicted_id[0].numpy()[0]] = 1
					#last_expected_np[i, last_predicted_id[i]] = 1
					
				
				if self.using_cuda:
					return_vector = Variable(torch.FloatTensor(last_expected_np).cuda())
				else:
					return_vector = Variable(torch.FloatTensor(last_expected_np))
			#######################
			# 
			# y(t-1) = GRU_output
			# h(t-1) = GRU_hidden
			# Ct	 = multiplied_mat
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
			return_vector = GRU_output.view(current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN)
			
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

			# pdb.set_trace()
			##########################################################
			######### COVERAGE #######################################
			##########################################################
			# This is a MLP Which receive 3 input:
			# Beta matrix: << Still have no idea how to implement this though :'(
			# FCN result
			# h(t-1): current hidden state of GRU


			# Get Input from h(t - 1)	
			from_h = self.Coverage_MLP_From_H(GRU_hidden.view(current_tensor_shape[0], self.gru_hidden_size))
			from_h = from_h.view(-1, 1, 1)
			from_h = from_h.repeat(1, 1, current_tensor_shape[2], current_tensor_shape[3])
			#from_h = self.Coverage_MLP_From_H(torch.squeeze(GRU_hidden, dim = 1))

			# New Approach
			FCN_Straight = FCN_Result.permute(0,2,3,1).contiguous()
			FCN_Straight = FCN_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], current_tensor_shape[1])
			from_a = self.Coverage_MLP_From_A(FCN_Straight)
			from_a = from_a.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
			# --
			F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
			#F_Straight = F_.transpose(1,3).contiguous()
			F_Straight = F_.permute(0,2,3,1).contiguous()
			F_Straight = F_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.Q_height)
			from_b = self.Coverage_MLP_From_Beta(F_Straight)
			from_b = from_b.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
			
			#---------------
			#from_a = from_a - (
                        #    from_a.norm()+
                        #    from_h.norm()*current_tensor_shape[2] * current_tensor_shape[3] * self.va_len
                        #) * torch.unsqueeze(beta_mat, dim=1)  + from_h.repeat(1, current_tensor_shape[2] * current_tensor_shape[3] * self.va_len).view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])

			#from_a = from_a + from_b + from_h.repeat(1, current_tensor_shape[2] * current_tensor_shape[3]).view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
			from_a = from_a + from_b + from_h
			#---------------


			#alpha_mat = torch.squeeze(from_a, dim = 1)
			#print (alpha_mat.cpu().data.numpy())
			
		
			
			# update alpha matrix
			#for batch_index in range(current_tensor_shape[0]):
			#	for i in range(current_tensor_shape[2]):
			#		for j in range(current_tensor_shape[3]):
			#			
			#			#t = torch.transpose(alpha_mat.view(3,4),0,1)
			#		# this is a_i vector
			#			temp_tensor = Variable(torch.unsqueeze(FCN_Result.data[batch_index,:,i,j], dim = 0))
			#			
			#			# get input from FCN_Result
			#			from_a = self.Coverage_MLP_From_A(temp_tensor)
			#
			#			# Assign pixel by pixel to alpha matrix
			#			alpha_mat.data[batch_index][i][j] = from_h.data[batch_index][0] + from_a.data[0][0]
			#			#pdb.set_trace()
			#
			#			print (alpha_mat.data.numpy().shape)
			
			#alpha_mat = alpha_mat.view(1,16, 32)
			alpha_mat = F.tanh(from_a)
			
			## Va fix ##

			#alpha_straight = alpha_mat.transpose(1,3).contiguous()
			alpha_straight = alpha_mat.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.va_len)
			alpha_mat = self.Va_fully_connected(alpha_straight)

			alpha_mat = alpha_mat.transpose(0,1).contiguous().view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
			


			alpha_mat = self.alpha_softmax(alpha_mat.view(current_tensor_shape[0], 512)).view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
#			pdb.set_trace()
			self.print_alpha_mat.append(alpha_mat.cpu().data.numpy())
			self.alpha_mat = torch.cat([self.alpha_mat, torch.unsqueeze(alpha_mat, dim = 1)], 1)
			
		#return torch.unsqueeze(return_tensor, dim = 1)
		# Returnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ! after a long long way :'(((((

		
		return return_tensor, self.alpha_mat




	def forwardTest(self, x, beam_size = 10):

		#print (self.training)

		####################################################################
		################ FCN BLOCK #########################################
		####################################################################

		x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
		x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
		x = F.relu(self.conv1_3_bn(self.conv1_3(x)))
		x = F.relu(self.conv1_4_bn(self.conv1_4(x)))
		
		x = self.pool_1(x)
		
		x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
		x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
		x = F.relu(self.conv2_3_bn(self.conv2_3(x)))
		x = F.relu(self.conv2_4_bn(self.conv2_4(x)))
		
		x = self.pool_2(x)
		
		x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
		x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
		x = F.relu(self.conv3_3_bn(self.conv3_3(x)))
		x = F.relu(self.conv3_4_bn(self.conv3_4(x)))
		
		x = self.pool_3(x)
		
		x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
		x = self.conv4_1_drop(x)
		x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
		x = self.conv4_2_drop(x)
		x = F.relu(self.conv4_3_bn(self.conv4_3(x)))
		x = self.conv4_3_drop(x)
		x = F.relu(self.conv4_4_bn(self.conv4_4(x)))
		x = self.conv4_4_drop(x)
		
		FCN_Result = self.pool_4(x)
		
		# Shape of FCU result: normally: (batchsize, 128, 16, 32)
		current_tensor_shape = FCN_Result.cpu().data.numpy().shape
		num_of_block = current_tensor_shape[2] * current_tensor_shape[3]
		################ DEFINITION ########################################
		
		start_vector = numpy.zeros((current_tensor_shape[0],1,NetWorkConfig.NUM_OF_TOKEN))
		start_vector[:,0,self.word_to_id['<s>']] = 1

		if self.using_cuda:
			#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
			GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], self.gru_hidden_size).cuda().zero_())
			# Init return tensor (the prediction of mathematical Expression)
			return_tensor = Variable(torch.FloatTensor(start_vector).cuda(), requires_grad=True)
			#Init Alpha and Beta Matrix
			alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).cuda().fill_(1.0 / float(num_of_block)), requires_grad=True)
			beta_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).cuda().zero_(), requires_grad=True)
		else:
			#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
			GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], self.gru_hidden_size).zero_())
			# Init return tensor (the prediction of mathematical Expression)
			return_tensor = Variable(torch.FloatTensor(start_vector), requires_grad=True)
			#Init Alpha and Beta Matrix
			alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).fill_(1.0 / float(num_of_block)), requires_grad=True)
			beta_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]).zero_(), requires_grad=True)
                #pdb.set_trace()
		####################################################################
		################ GRU BLOCK #########################################
		####################################################################

		
		
		################### START GRU ########################

		
		# insert_index = 1
		
		# Init the first vector in return_tensor: It is the <s> token
#		return_tensor.data[:, 0, self.word_to_id['<s>']] = 1

		# Get last predicted symbol: This will be used for GRU's input
		return_vector = torch.squeeze(return_tensor, dim = 1)
		#GRU_output = Variable(return_tensor.data[:, 0, :])
		
		####################################################################
		################ GRU ITERATION #####################################
		####################################################################
		
		# True : expand beam tree
		# False: Evaluate and set result
		is_predicting = True


		for RNN_iterate in range (2 * NetWorkConfig.MAX_TOKEN_LEN - 2):

			if is_predicting:
				is_predicting = False
				# Clone of FCN_Result: We will use this for generating Ct Vector || Deprecated - We use another approach now!
				multiplied_mat = FCN_Result.clone()

				expanded_alpha_mat = alpha_mat.view(current_tensor_shape[0], 1, current_tensor_shape[2], current_tensor_shape[3])
				expanded_alpha_mat = expanded_alpha_mat.repeat(1, current_tensor_shape[1], 1, 1)
				multiplied_mat = multiplied_mat * expanded_alpha_mat
				
				# Sum all vector after element-wise multiply to get Ct
				if NetWorkConfig.CURRENT_MACHINE == 0:
					multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 2)
					multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 3)
				else:
					multiplied_mat = torch.sum(multiplied_mat, dim = 2)
					multiplied_mat = torch.sum(multiplied_mat, dim = 3)
				#multiplied_mat = self.testnn(multiplied_mat.view(current_tensor_shape[0], 65536))
				
				multiplied_mat = multiplied_mat.view(current_tensor_shape[0], 128)
				
				
				########################################################################################
				################### GRU SECTION ########################################################
				########################################################################################

				#--------------------------------------------------------------------
				 
				if self.training == True:
					
					last_expected_id = self.GT[:, int(RNN_iterate / 2)]
					last_expected_np = numpy.zeros((current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN))
					for i in range(current_tensor_shape[0]):
						last_expected_np[i, last_expected_id[i]] = 1
					
					if self.using_cuda:
						return_vector = Variable(torch.FloatTensor(last_expected_np).cuda())
					else:
						return_vector = Variable(torch.FloatTensor(last_expected_np))
				else:


					last_predicted_id = return_vector.max(1)[1].data

					last_expected_id = self.GT[:, int(RNN_iterate / 2)]
					
					
					#if last_predicted_id[0] != 19:
					#	print('---------------------')

					last_expected_np = numpy.zeros((current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN))
						

								
					for i in range(current_tensor_shape[0]):
					
						#if last_predicted_id[0] != 19:
						last_expected_np[i, last_predicted_id[i]] = 1
						#last_expected_np[i, last_predicted_id[i]] = 1
						
					
					if self.using_cuda:
						return_vector = Variable(torch.FloatTensor(last_expected_np).cuda())
					else:
						return_vector = Variable(torch.FloatTensor(last_expected_np))
				#######################
				# 
				# y(t-1) = GRU_output
				# h(t-1) = GRU_hidden
				# Ct	 = multiplied_mat
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

				return_vector = GRU_output.view(current_tensor_shape[0], NetWorkConfig.NUM_OF_TOKEN)
				

				#print(self.id_to_word[return_vector.max(1)[1].data.numpy()[0,0]])

				#print(return_vector.max(1)[1].data)
##return_tensor = torch.cat([return_tensor, torch.unsqueeze(return_vector, dim = 1)], 1)
				beta_mat = beta_mat + alpha_mat
				
				##########################################################
				######### COVERAGE #######################################
				##########################################################
				# This is a MLP Which receive 3 input:
				# Beta matrix: << Still have no idea how to implement this though :'(
				# FCN result
				# h(t-1): current hidden state of GRU


				# Get Input from h(t - 1)
				from_h = self.Coverage_MLP_From_H(GRU_hidden.view(current_tensor_shape[0], self.gru_hidden_size))
				#from_h = self.Coverage_MLP_From_H(torch.squeeze(GRU_hidden, dim = 1))

				# New Approach
				FCN_Straight = FCN_Result.permute(0,2,3,1).contiguous()
				FCN_Straight = FCN_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], current_tensor_shape[1])
				from_a = self.Coverage_MLP_From_A(FCN_Straight)
				from_a = from_a.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
				# --
				F_ = self.conv_Q_beta(torch.unsqueeze(beta_mat, dim = 1)) #(13)
				#F_Straight = F_.transpose(1,3).contiguous()
				F_Straight = F_.permute(0,2,3,1).contiguous()
				F_Straight = F_Straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.Q_height)
				from_b = self.Coverage_MLP_From_Beta(F_Straight)
				from_b = from_b.transpose(0,1).contiguous().view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
				
				#---------------
				
				from_a = from_a + from_b + from_h.repeat(1, current_tensor_shape[2] * current_tensor_shape[3] * self.va_len).view(current_tensor_shape[0], self.va_len, current_tensor_shape[2], current_tensor_shape[3])
				#---------------


				alpha_mat = torch.squeeze(from_a, dim = 1)


				alpha_mat = F.tanh(from_a)

				## Va fix ##
				alpha_straight = alpha_mat.contiguous()
				#alpha_straight = alpha_straight.transpose(1,3).contiguous()
				
				
				alpha_straight = alpha_straight.view(current_tensor_shape[0] * current_tensor_shape[2] * current_tensor_shape[3], self.va_len)
				alpha_mat = self.Va_fully_connected(alpha_straight)

				alpha_mat = alpha_mat.transpose(0,1).contiguous().view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
				


				alpha_mat = self.alpha_softmax(alpha_mat.view(current_tensor_shape[0], 512)).view(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3])
				self.attention_list.append(alpha_mat.data.numpy())


			else:
				is_predicting = True

				# Clone of FCN_Result: We will use this for generating Ct Vector || Deprecated - We use another approach now!
				multiplied_mat = FCN_Result.clone()

				
				expanded_alpha_mat = alpha_mat.view(current_tensor_shape[0], 1, current_tensor_shape[2], current_tensor_shape[3])
				expanded_alpha_mat = expanded_alpha_mat.repeat(1, current_tensor_shape[1], 1, 1)
				multiplied_mat = multiplied_mat * expanded_alpha_mat
				
				# Sum all vector after element-wise multiply to get Ct
				if NetWorkConfig.CURRENT_MACHINE == 0:
					multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 2)
					multiplied_mat = torch.sum(multiplied_mat, keepdim=True, dim = 3)
				else:
					multiplied_mat = torch.sum(multiplied_mat, dim = 2)
					multiplied_mat = torch.sum(multiplied_mat, dim = 3)
				#multiplied_mat = self.testnn(multiplied_mat.view(current_tensor_shape[0], 65536))
				
				multiplied_mat = multiplied_mat.view(current_tensor_shape[0], 128)
				
				
				########################################################################################
				################### GRU SECTION ########################################################
				########################################################################################

				#--------------------------------------------------------------------
				 

				#last_predicted_id = return_vector.max(1)[1].data

				
				
				last_return_vector = return_vector

				#last_predicted_id = return_vector.data.numpy()[0]
				
				
				
				last_predicted_id = return_vector.data.numpy()[0]

				
				
				last_predicted_Candidates = numpy.argsort(last_predicted_id)[::-1][:beam_size]

					
				
				###########################################333
				######### BEAM MULTI #######################3#
				###########################################333



				last_expected_np = numpy.zeros((current_tensor_shape[0] * beam_size, NetWorkConfig.NUM_OF_TOKEN))
				multiplied_mat = multiplied_mat.repeat(beam_size, 1)
				GRU_hidden_eval = GRU_hidden.repeat(beam_size, 1)


								
				for i in range(current_tensor_shape[0] * beam_size):
					
						#if last_predicted_id[0] != 19:
					last_expected_np[i, last_predicted_Candidates[i]] = 1
						#last_expected_np[i, last_predicted_id[i]] = 1
				

				if self.using_cuda:
					return_vector = Variable(torch.FloatTensor(last_expected_np).cuda())
				else:
					return_vector = Variable(torch.FloatTensor(last_expected_np))
					
					
				#######################
				# 
				# y(t-1) = GRU_output
				# h(t-1) = GRU_hidden
				# Ct	 = multiplied_mat
				embedded = self.embeds_temp(return_vector)


				zt = self.FC_Wyz(embedded) + self.FC_Uhz(GRU_hidden_eval) + self.FC_Ccz(multiplied_mat) # equation (4) in paper
				zt = F.sigmoid(zt)
				
				rt = self.FC_Wyr(embedded) + self.FC_Uhr(GRU_hidden_eval) + self.FC_Ccr(multiplied_mat) # (5)
				rt = F.sigmoid(rt)
				
				ht_candidate = self.FC_Wyh(embedded) + self.FC_Urh(rt * GRU_hidden_eval) + self.FC_Ccz(multiplied_mat) #6
				ht_candidate = F.tanh(ht_candidate)
				
				GRU_hidden_eval = (1 - zt) * GRU_hidden_eval + zt * ht_candidate # (7)
				
				GRU_output_temp = self.FC_Wo(embedded + self.FC_Wh(GRU_hidden_eval) + self.FC_Wc(multiplied_mat)) 
				
			
				return_vector_temp = GRU_output_temp.view(current_tensor_shape[0] * beam_size, NetWorkConfig.NUM_OF_TOKEN)
				
				##################################
				####### GETTING MAX ##############
				##################################

				#print('---------------------')
				#print (return_vector_temp.data.numpy())
				#print('-//////////////////////////////////')
				#print (last_return_vector.data.numpy())

				np_return_vector_temp = return_vector_temp.data.numpy()
				np_last_return_vector = last_return_vector.data.numpy()[0]
				
				
				for i in range(beam_size):
					np_return_vector_temp[i] = np_return_vector_temp[i] + np_last_return_vector[last_predicted_Candidates[i]]
				
				#for candidate in last_predicted_Candidates:
				#	np_return_vector_temp = np_return_vector_temp + np_last_return_vector

				max_values = []

				for candidate in np_return_vector_temp:
					max_values.append(max(candidate))
					
					

				choose = numpy.argmax(max_values)

				
				if self.using_cuda:
					return_vector = Variable(torch.FloatTensor(return_vector).cuda())
				else:
					return_vector = Variable(torch.FloatTensor(return_vector))

				return_vector = return_vector.view(1, NetWorkConfig.NUM_OF_TOKEN)

				return_tensor = torch.cat([return_tensor, torch.unsqueeze(return_vector, dim = 1)], 1)
				

		#return torch.unsqueeze(return_tensor, dim = 1)
		# Returnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ! after a long long way :'(((((
		return return_tensor


	############## UTILS ######################33
	def createVector(self, v, batch, toklen):
		z = numpy.zeros((batch, toklen))
		for i in range(batch):
			z[i, v[i]] = 1
		return z