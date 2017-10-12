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
testnet.loadModelFromFile('model/mymodel.mdl')

if using_cuda and cuda_avail:
	testnet.model.cuda()
	testnet.setCudaState()

#############################################
######### TRAINING AND TESTING ##############
#############################################

iteration = 0

loader.init(NC.DATASET_PATH)

record_count = 0
sum_loss_exp = 0
sum_loss_dist = 0
sum_loss_ecl = 0
perfect = 0
while True:
	train_data = loader.getNextDataset(batch_size)

	if train_data == False:
		break

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

	testnet.setData(train_loader, 0)

		#testnet.train(epoch + 1)
	#print(testnet.test(1, batch_size = batch_size))

	print('--------------------------------------')

	loss_w, loss_l, loss_e = testnet.testAll(batch_size = batch_size)


	if loss_w < 0.1:
		perfect = perfect + 1

	sum_loss_exp = sum_loss_exp +  loss_w
	sum_loss_dist = sum_loss_dist +  loss_l
	sum_loss_ecl = sum_loss_ecl + loss_e

	print(loss_w)
	print(loss_l)
	print(loss_e)


	iteration = iteration + 1
	
	break
	


print ('-------LOSS---------')
print (perfect)
print (sum_loss_exp/ float(iteration))
print (sum_loss_dist/ float(iteration))
print (sum_loss_ecl/ float(iteration))

f = open('lr.txt', 'w')
f.write(str(perfect))
f.write('--')
f.write(str(sum_loss_exp/ float(iteration)))
f.write('--')
f.write(str(sum_loss_dist/ float(iteration)))
f.write('--')
f.write(str(sum_loss_ecl/ float(iteration)))
f.close()


#testnet.loadModelFromFile('model/version5.mdl')
#t

