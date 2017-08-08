from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy

import NetWorkConfig

class WAP(nn.Module):
	def __init__(self):
		super(WAP, self).__init__()
		self.conv1_1 = nn.Conv2d(1, 32, 3, stride=1,padding=1)
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


		self.Coverage_MLP_From_H = nn.Linear(128, 1)
		self.Coverage_MLP_From_A = nn.Linear(128, 1)

		self.alpha_softmax = torch.nn.Softmax()

		self.max_output_len  = NetWorkConfig.MAX_TOKEN_LEN



	def setWordMaxLen(self, max_len):
		self.max_output_len = max_len

	def forward(self, x):

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

		#print (FCN_Result.data.numpy().shape)
		################### END FCN ##########################

		current_tensor_shape = FCN_Result.data.numpy().shape


		################### START GRU ########################
		#GRU_hidden = Variable(torch.FloatTensor(current_tensor_shape[0], 128))
		GRU_hidden = Variable(torch.rand(current_tensor_shape[0], 128))

		# remember to fix dim//// symbol x index // And assign Starting
		return_tensor = Variable(torch.FloatTensor(current_tensor_shape[0], 1, NetWorkConfig.NUM_OF_TOKEN).zero_(), requires_grad=True)

		alpha_mat = Variable(torch.FloatTensor(current_tensor_shape[0], current_tensor_shape[2], current_tensor_shape[3]), requires_grad=True)

		for RNN_iterate in range (self.max_output_len - 1):

			multiplied_mat = FCN_Result.clone()

			for batch_index in range(current_tensor_shape[0]):
				for i in range (current_tensor_shape[1]):
					multiplied_mat.data[batch_index][i] = multiplied_mat.data[batch_index][i] * alpha_mat.data[batch_index]

			multiplied_mat = torch.sum(multiplied_mat, dim = 2)
			multiplied_mat = torch.sum(multiplied_mat, dim = 3)

			multiplied_mat = multiplied_mat.view(current_tensor_shape[0], 128)

			GRU_hidden = self.grucell(multiplied_mat, GRU_hidden)
			#print (GRU_output.data.numpy().shape)

			GRU_output = self.post_gru(GRU_hidden)
			

			GRU_output = torch.unsqueeze(GRU_output, dim = 1)

			
			
			return_tensor = torch.cat([return_tensor, F.log_softmax(GRU_output)], 1)

			
			
			#print (GRU_hidden.data.numpy().shape)

			##########################################################
			######### COVERAGE #######################################
			##########################################################


			from_h = self.Coverage_MLP_From_H(torch.squeeze(GRU_hidden, dim = 1))
			
			
			
			#from_a = self.Coverage_MLP_From_A(FCN_Result.data[batch_index,:,i,j])

			

			for batch_index in range(current_tensor_shape[0]):
				for i in range(current_tensor_shape[2]):
					for j in range(current_tensor_shape[3]):
						pass
						
						temp_tensor = Variable(torch.unsqueeze(FCN_Result.data[batch_index,:,i,j], dim = 0))
						
						
						
						from_a = self.Coverage_MLP_From_A(temp_tensor)

						
						
						alpha_mat.data[batch_index][i][j] = from_h.data[batch_index][0] + from_a.data[0][0]

						#print(alpha_mat.data.numpy().shape)
				
			
				
			alpha_mat = F.tanh(alpha_mat)
			alpha_mat = self.alpha_softmax(alpha_mat.view(current_tensor_shape[0], 512)).view(current_tensor_shape[0], 16, 32)

			
		#print (return_tensor.data.numpy().shape)	

		#return torch.unsqueeze(return_tensor, dim = 1)
		return return_tensor

