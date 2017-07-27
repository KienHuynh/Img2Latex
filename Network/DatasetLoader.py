from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy
import struct

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
		

		Tensor_train = self.getTensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_target.astype(numpy.long)))
		Tensor_test = self.getTensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_target.astype(numpy.long)))

		return Tensor_train, Tensor_test

	def getTensorDataset(self, input_data, target):
		return torch.utils.data.TensorDataset(input_data, target)

#loader = Loader()
#trainn, test = loader.generateTensorDatasetFromMNISTFolder('data/raw/')
#train_loader = torch.utils.data.DataLoader(trainn, batch_size=100, shuffle=False)

