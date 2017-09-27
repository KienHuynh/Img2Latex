from __future__ import print_function
import torch

import DatasetLoader as DL
import CNNNetwork as testnetwork
import NetWorkConfig as NC

import os
import numpy as np
batch_size = NC.BATCH_SIZE

#############################################
######### TESTING HARDWARE ##################
#############################################

using_cuda = False
cuda_avail = torch.cuda.is_available()

#############################################
######### DATA LOADING ######################
#############################################

#loader = DL.Loader()
#train, test = loader.generateTensorDatasetFromMNISTFolder('../data/MNIST/')
#train, test = loader.generateTensorDatasetFromCROHMEBinary('../data/CROHME/Binary/CROHMEBLOCK_Data1.npy', '../data/CROHME/Binary/CROHMEBLOCK_Target1.npy', '../data/CROHME/Binary/CROHMEBLOCK_Data.npy', '')
#train, test = loader.generateTensorDatasetFromCROHMEBinary('../data/CROHME/Binary/CROHMEBLOCK_Data_M.npy', '../data/CROHME/Binary/CROHMEBLOCK_Target_M.npy', '../data/CROHME/Binary/CROHMEBLOCK_Data.npy', '')

#train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
############

loader = DL.loadDatasetFileByFile()




#############
testnet = testnetwork.TestingNetwork()


if using_cuda and cuda_avail:
	testnet.model.cuda()
	testnet.setCudaState()

#############################################
######### TRAINING AND TESTING ##############
#############################################

br = 0

for epoch in range(NC.EPOCH_COUNT):
	loader.init(NC.DATASET_PATH)
	#print(NC.EPOCH_COUNT)
	#break 
	#testnet.ite = 0
	
	while True:
		train_data = loader.getNextDataset(batch_size)

		if train_data == False:
			break

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

		testnet.setData(train_loader, 0)

		testnet.train(epoch + 1)
	
#		with open("params.txt", "wb") as f:
#			for batch_id in range (NC.BATCH_SIZE):
				
#				np.savetxt(f,testnet.model.conv1_1.weight.data[batch_id,:,:].numpy(),fmt='%-1.7G',footer='====')
#				print(testnet.model.conv1_1.weight.data[batch_id,:,:].numpy())
		
		#breakbreak 

#		br = br + 1
#		if br == 3:
#			break
		#break
	if epoch%10==9:
		try:
			os.mkdir(NC.MODEL_FOLDER)
		except:
			pass
		testnet.saveModelToFile(NC.MODEL_FOLDER + 'version_'+str(epoch)+'.mdl')		
#print(testnet.model.attention_list[len(testnet.model.attention_list)-1])

#testnet.loadModelFromFile('model/version1.mdl')
#testnet.test()


testnet.saveModelToFile('model/version15_09.mdl')
