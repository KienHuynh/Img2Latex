from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy
import struct
import cv2

#train = torch.utils.data.TensorDataset(a, b)
#train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=False)

#for data, target in train_loader:
#	print (data)

class Loader:
	def __init__(self):
		print ('Loader init')


	def read_MNISTfile(self, filename):
		with open(filename, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
			return numpy.fromstring(f.read(), dtype=numpy.uint8).reshape(shape)

	def generateTensorDatasetFromMNISTFolder(self, path):
		train_set = self.read_MNISTfile(path + 'train-images-idx3-ubyte')[:,numpy.newaxis]
		train_target = self.read_MNISTfile(path + 'train-labels-idx1-ubyte')
		test_set = self.read_MNISTfile(path + 't10k-images-idx3-ubyte')[:,numpy.newaxis]
		test_target = self.read_MNISTfile(path + 't10k-labels-idx1-ubyte')
		
		print (train_set.shape)

		Tensor_train = self.getTensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_target.astype(numpy.double)))
		Tensor_test = self.getTensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_target.astype(numpy.double)))

		return Tensor_train, Tensor_test

	def read_CROHMEFolder(self, path):
		return_data = []
		print (return_data)


	def generateTensorDatasetFromCROHMEFolder(self, path):
		train_data = self.read_CROHMEFolder(path)
		pass

	def getTensorDataset(self, input_data, target):
		return torch.utils.data.TensorDataset(input_data, target)


loader = Loader()
loader.generateTensorDatasetFromCROHMEFolder('../data/CROHME/img/')
#trainn, test = loader.generateTensorDatasetFromMNISTFolder('../data/MNIST/')
#train_loader = torch.utils.data.DataLoader(trainn, batch_size=100, shuffle=False)

#loader = DL.Loader()
#train, test = loader.generateTensorDatasetFromMNISTFolder('../data/MNIST/')