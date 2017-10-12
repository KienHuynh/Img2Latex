from __future__ import print_function
import torch

import DatasetLoader as DL
import CNNNetwork as testnetwork
import NetWorkConfig as NC

import os
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


############# version15_09.mdl
testnet = testnetwork.TestingNetwork()
#testnet.loadModelFromFile('./model/train/version_1210_14.mdl')
torch.manual_seed(2017)
if using_cuda and cuda_avail:
	if hasattr(testnet.model, 'cuda'):
		testnet.model.cuda()
	else:
		testnet.model
	#testnet.model.cuda()
	testnet.setCudaState()
	torch.cuda.manual_seed(2017)

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
		train_data, attendGT= loader.getNextDataset(batch_size)

		if train_data == False:
#			pdb.set_trace()
			break

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
#		pdb.set_trace()
		testnet.setData(train_loader,attendGT, 0)

		testnet.train(epoch + 1)
		
		#breakbreak 

#		br = br + 1
#		if br == 3:
#			break
		break
	if epoch%2 == 9:
		for i in range(len(testnet.model.print_alpha_mat)):
			plt.imshow(testnet.model.print_alpha_mat[i][0,:], cmap='gray', interpolation='nearest')
			plt.savefig('figures/tmp0_%03d.png' % i)
			plt.clf()
	if epoch%5==4:
		try:
			os.mkdir(NC.MODEL_FOLDER)
		except:
			pass
		testnet.saveModelToFile(NC.MODEL_FOLDER + 'version_1210_'+str(epoch)+'.mdl')	
		testnet.saveLearningRate()	

#testnet.loadModelFromFile('model/version1.mdl')
#testnet.test()

#print('alpha_mat',testnet.model.alpha_mat[88].shape)

testnet.saveModelToFile('model/version26_09.mdl')
testnet.saveLearningRate()	
