from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy

import NetWorkConfig
import getGT
import pdb

class WAP(nn.Module):
	def __init__(self):
		super(WAP, self).__init__()
		self.conv1_1 = nn.Conv2d(9, 32, 3, stride=1,padding=1)
		self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_4 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		
		self.pool_1 = nn.MaxPool2d(2, stride=2)
		
		self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
		self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		
		self.pool_2 = nn.MaxPool2d(2, stride=2)
		self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		
		self.pool_3 = nn.MaxPool2d(2, stride=2)
		
		self.conv4_1 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
		self.conv4_2 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_3 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_4 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		
		self.pool_4 = nn.MaxPool2d(2, stride=2)


		###############################################
		########### ATTENTION #########################
		###############################################


		###############################################
		########### GRU ###############################
		###############################################

		#Outputs: output, h_n
		self.gru = nn.GRU(input_size  = 128, hidden_size  = 128)
		self.grucell = nn.GRUCell(128, 128)
		self.post_gru = nn.Linear(128, NetWorkConfig.NUM_OF_TOKEN)
		
		self.Out_to_hidden_GRU = nn.Linear(NetWorkConfig.NUM_OF_TOKEN, 128)
		
		self.Coverage_MLP_From_H = nn.Linear(128, 1)
		self.Coverage_MLP_From_A = nn.Linear(128, 1)
		
		self.alpha_softmax = torch.nn.Softmax()
		
		self.max_output_len  = NetWorkConfig.MAX_TOKEN_LEN
		
	def setGroundTruth(self, GT):
		self.GT = GT
		
	def setWordMaxLen(self, max_len):
		self.max_output_len = max_len
		
	def forward(self, x):

		####################################################################
		################ FCN BLOCK #########################################
		####################################################################
		
		
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.relu(self.conv1_3(x))
		x = F.relu(self.conv1_4(x))
		
		x = F.relu(self.pool_1(x))
		
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.relu(self.conv2_3(x))
		x = F.relu(self.conv2_4(x))
		
		x = F.relu(self.pool_2(x))
		
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = F.relu(self.conv3_4(x))
		
		x = F.relu(self.pool_3(x))
		
		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = F.relu(self.conv4_4(x))
		
		
		FCN_Result = F.relu(self.pool_4(x))

		####################################################################
		################ GRU BLOCK #########################################
		####################################################################

		# Shape of FCU result: normally: (batchsize, 128, 16, 32)
		current_tensor_shape = FCN_Result.data.numpy().shape


		################### START GRU ########################

		# Init Gru hidden Randomly
		#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
		GRU_hidden = Variable(torch.rand(current_tensor_shape[0], 128))

		# Init return tensor (the prediction of mathematical Expression)
		return_tensor = Variable(torch.FloatTensor(current_tensor_shape[0], 1, NetWorkConfig.NUM_OF_TOKEN).zero_(), requires_grad=True)

		# Init the first vector in return_tensor: It is the <s> token
		return_tensor.data[:, 0, getGT.word_to_id['<s>']] = 1

		# Get last predicted symbol: This will be used for GRU's input
		last_y = torch.squeeze(return_tensor, dim = 1)

		#Init Alpha and Beta Matrix
		alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]), requires_grad=True)
		beta_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]), requires_grad=True)
#		pdb.set_trace()
		####################################################################
		################ GRU ITERATION #####################################
		####################################################################
		
		for RNN_iterate in range (self.max_output_len - 1):

			# Clone of FCN_Result: We will use this for generating Ct Vector
			multiplied_mat = FCN_Result.clone()

			# Element-wise multiply between alpha and FCN_Result
			for batch_index in range(current_tensor_shape[0]):
				for i in range (current_tensor_shape[1]):
					multiplied_mat.data[batch_index][i] = multiplied_mat.data[batch_index][i] * alpha_mat.data[batch_index]

			# Sum all vector after element-wise multiply to get Ct
			multiplied_mat = torch.sum(multiplied_mat, dim = 2)
			multiplied_mat = torch.sum(multiplied_mat, dim = 3)
			multiplied_mat = multiplied_mat.view(current_tensor_shape[0], 128)

			# Generating GRU's input, this is neuron from y(t-1) - from_last_output
			# Input of GRU Cell consist of y(t-1), h(t-1) and Ct (and Some gate in GRU Cell ... I think pytorch will handle itself)
			from_last_output = self.Out_to_hidden_GRU(last_y)

			# Run GRUcell, Calculate h(t) - GRU_hidden
			# y(t-1) = from_last_output
			# h(t-1) = GRU_hidden
			# Ct	 = multiplied_mat
			GRU_hidden = self.grucell(multiplied_mat + from_last_output, GRU_hidden)
			#print (GRU_output.data.numpy().shape)

			# From Gru hidden, we calculate the GRU's output vector (or the prediction of next symbol) 
			GRU_output = self.post_gru(GRU_hidden)
			# Update last prediction
			last_y = GRU_output


			# Apply softmax to prediction vector and concatenate to return_tensor
			GRU_output = torch.unsqueeze(GRU_output, dim = 1)
			return_vector = Variable(torch.squeeze(GRU_output.data, dim = 1))

			# return_vector = F.softmax(Variable(torch.squeeze(GRU_output.data, dim = 1)))
			# return_tensor = torch.cat([return_tensor, torch.unsqueeze(F.softmax(Variable(torch.squeeze(GRU_output.data, dim = 1))), dim = 1)], 1)
			return_tensor = torch.cat([return_tensor, torch.unsqueeze(return_vector, dim = 1)], 1)
			# pdb.set_trace()
			##########################################################
			######### COVERAGE #######################################
			##########################################################
			# This is a MLP Which receive 3 input:
			# Beta matrix: << Still have no idea how to implement this though :'(
			# FCN result
			# h(t-1): current hidden state of GRU

			# Get Input from h(t - 1)
			from_h = self.Coverage_MLP_From_H(torch.squeeze(GRU_hidden, dim = 1))
			


			for batch_index in range(current_tensor_shape[0]):
				


			# update alpha matrix
			for batch_index in range(current_tensor_shape[0]):
				for i in range(current_tensor_shape[2]):
					for j in range(current_tensor_shape[3]):
						
						#t = torch.transpose(alpha_mat.view(3,4),0,1)
						# this is a_i vector
						temp_tensor = Variable(torch.unsqueeze(FCN_Result.data[batch_index,:,i,j], dim = 0))
						
						# get input from FCN_Result
						from_a = self.Coverage_MLP_From_A(temp_tensor)

						# Assign pixel by pixel to alpha matrix
						alpha_mat.data[batch_index][i][j] = from_h.data[batch_index][0] + from_a.data[0][0]
						#pdb.set_trace()
			
			
			
			alpha_mat = F.tanh(alpha_mat)
			alpha_mat = self.alpha_softmax(alpha_mat.view(current_tensor_shape[0], 512)).view(current_tensor_shape[0], 16, 32)

			
		#return torch.unsqueeze(return_tensor, dim = 1)
		# Returnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ! after a long long way :'(((((
		return return_tensor

