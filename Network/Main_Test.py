from __future__ import print_function
import torch

import DatasetLoader as DL
import CNNNetwork as testnetwork
import NetWorkConfig as NC
batch_size = NC.BATCH_SIZE

#############################################
######### TESTING HARDWARE ##################
#############################################

using_cuda = False
cuda_avail = torch.cuda.is_available()

#############################################
######### DATA LOADING ######################
#############################################


loader = DL.loadDatasetFileByFile()




#############
testnet = testnetwork.TestingNetwork()
testnet.loadModelFromFile('model/version1.mdl')

if using_cuda and cuda_avail:
	testnet.model.cuda()
	testnet.setCudaState()

#############################################
######### TRAINING AND TESTING ##############
#############################################

br = 0

for epoch in range(NC.EPOCH_COUNT):
	loader.init(NC.DATASET_PATH)
	testnet.ite = 0
	while True:
		train_data = loader.getNextDataset(batch_size)

		if train_data == False:
			break

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

		testnet.setData(train_loader, train_loader)

		testnet.test(epoch)
		
		break

#testnet.loadModelFromFile('model/version1.mdl')
#testnet.test()


#testnet.saveModelToFile('model/version1.mdl')
