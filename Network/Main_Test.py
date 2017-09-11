from __future__ import print_function
import torch

import DatasetLoader as DL
import CNNNetwork as testnetwork
import NetWorkConfig as NC
batch_size = NC.BATCH_SIZE

#############################################
######### TESTING HARDWARE ##################
#############################################

using_cuda = True
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
testnet.loadModelFromFile('model/official_ver_1.mdl')

if using_cuda and cuda_avail:
	testnet.model.cuda()
	testnet.setCudaState()

#############################################
######### TRAINING AND TESTING ##############
#############################################

br = 0

loader.init(NC.DATASET_PATH)
while True:
	train_data = loader.getNextDataset(batch_size)

	if train_data == False:
		break

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

	testnet.setData(train_loader, 0)

		#testnet.train(epoch + 1)
	testnet.test()


	br = br + 1
	if br == 49:
		break


#testnet.loadModelFromFile('model/version5.mdl')
#t
