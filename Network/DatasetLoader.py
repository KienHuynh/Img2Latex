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

from random import shuffle
import os
import random

import sys
sys.path.insert(0, 'parser')
import CROHMEParser

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

	def read_CROHMEFile(self, path):
		return numpy.load(path)


	def generateTensorDatasetFromCROHMEBinary(self, pathtrain_data, pathtrain_target, pathtest_data, pathtest_target):
		train_set = numpy.load(pathtrain_data)
		test_set = numpy.load(pathtest_data)

		train_target = numpy.load(pathtrain_target)
		test_target = numpy.ones(len(test_set))
		
		print (train_set.shape)
		print (train_target.shape)
		Tensor_train = self.getTensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_target.astype(numpy.int64)))
		Tensor_test = self.getTensorDataset(torch.from_numpy(test_set), torch.from_numpy(test_target.astype(numpy.int64)))

		return Tensor_train, Tensor_test

	def generateTensorDatasetFromCROHMENumpy(self, train_set, train_target):
		Tensor_train = self.getTensorDataset(torch.from_numpy(train_set), torch.from_numpy(train_target.astype(numpy.int64)))
		return Tensor_train

	def getTensorDataset(self, input_data, target):
		return torch.utils.data.TensorDataset(input_data, target)

class loadDatasetFileByFile:
	def __init__(self, parse_result_path = './../data/ParseResult/'):
		self.loader = Loader()
		self.result_path = parse_result_path
		try:
			os.mkdir(parse_result_path)
		except:
			pass

	def getNextDataset(self, batch_size):
		to_parse_list = []
		
		#try:
		if True:
			while(len(to_parse_list)<batch_size):
#			for i in range(batch_size):
				if len(self.parent_path) == 0:
					print ('no more data')
					return False
				choose_index = random.randint(0, self.num_of_folder - 1)
				inkml_index = random.randint(0, self.folder_size[choose_index])
				
				# TODO Ngoc: Nen xoa cac file lg thi hon, dung co compare nay vi no se lam giam toc do train
				files = self.parent_path[choose_index] + self.inkml_list[choose_index][inkml_index]
				if not files.endswith('.lg'):
					to_parse_list.append((self.parent_path[choose_index] + self.inkml_list[choose_index][inkml_index], self.param_list[choose_index]))

				#to_parse_list.append((self.parent_path[choose_index] + self.inkml_list[choose_index][inkml_index], self.param_list[choose_index]))
				#print (to_parse_list)

				del self.inkml_list[choose_index][inkml_index]
				
				self.folder_size[choose_index] = self.folder_size[choose_index] - 1
				if self.folder_size[choose_index] == -1:
					self.num_of_folder = self.num_of_folder - 1
					del self.parent_path[choose_index]
					del self.inkml_list[choose_index]
					del self.folder_size[choose_index]
					del self.param_list[choose_index]

				
			#print (to_parse_list)

			dataset, target = CROHMEParser.ParseList(to_parse_list)

			#print (dataset.shape)
			#print (target.shape)

			dataset_dat = self.loader.generateTensorDatasetFromCROHMENumpy(dataset, target)
			
			return dataset_dat
		return False
		#except Exception as e:
		#	print ('Did you forget call \'init\'?')
		#	print (e)
		#	return False, False

	def init(self, path = "./../data/TrainINKML/"):
		self.parent_path, self.inkml_list, self.folder_size, self.param_list = self.getFileList(path)
		self.num_of_folder = len(self.parent_path)

	def getFileList(self, path = "./../data/TrainINKML/"):
		inkml_list = []
		parent_path = []
		folder_size = []
		param_list = []

		for root, dirs, files in os.walk(path, topdown=False):
			shuffle(files)
			inkml_list.append(files)
			parent_path.append(root + '/')
			folder_size.append(len(files) - 1)

		folder_size.pop()
		inkml_list.pop()
		parent_path.pop()

		for i in range(len(parent_path)): #(Scale factor)
			if 'expressmatch' in parent_path[i]:
				param_list.append((1,0))
			elif 'TEST' in parent_path[i]:
				param_list.append((0.5,0))
			elif 'HAMEX' in parent_path[i]:
				param_list.append((100,0))
			elif 'KAIST' in parent_path[i]:
				param_list.append((0.065,0))
			elif 'MathBrush' in parent_path[i]:
				param_list.append((0.04,0))
			else:
				param_list.append((0.8,0))

		return parent_path, inkml_list, folder_size, param_list


#z = loadDatasetFileByFile()
#z.init()
#aa = z.getNextDataset(10)
#print (aa)

#loader = Loader()
#loader.generateTensorDatasetFromCROHMEBinary('../data/CROHME/Binary/CROHMEBLOCK_Data_mini.npy', '../data/CROHME/Binary/CROHMEBLOCK_Target_mini.npy', '../data/CROHME/Binary/CROHMEBLOCK_Data.npy', '')
#trainn, test = loader.generateTensorDatasetFromMNISTFolder('../data/MNIST/')
#train_loader = torch.utils.data.DataLoader(trainn, batch_size=100, shuffle=False)

#loader = DL.Loader()
#train, test = loader.generateTensorDatasetFromMNISTFolder('../data/MNIST/')
