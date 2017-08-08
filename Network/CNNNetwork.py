from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy

import WAP

import NetWorkConfig

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):



		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

		x = x.view(-1, 320)
		#
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x)



class TestingNetwork:
	def __init__(self):
		print ('network init')

		################################################
		########## ATTRIBUTE INIT ######################
		################################################

		self.using_cuda = False
		self.batch_size = 64
		self.learning_rate = 0.01
		self.momentum = 0.5


		self.model = WAP.WAP()
		#self.model = Net()
		#self.model = FCN()
		
		self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

		self.NLLloss = nn.NLLLoss2d()

	def setCudaState(self, state = True):
		self.using_cuda = state

	def setData(self, train, test):
		self.train_loader = train
		self.test_loader = test

	def train(self, epoch):
		self.model.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):

			if self.using_cuda:
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data.float()), Variable(target.long().transpose(0,1))
			self.optimizer.zero_grad()
			output = self.model(data)

			

			output = output.view(NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)
			
			target = target.view(50)

			#print (type(output))
			print (target.data)
			print (target.data.numpy().shape)
			#target = torch.unsqueeze(target, dim = 1)

			print (output.data.numpy().shape)
			
			

			#loss = F.nll_loss(output, target)
			
			loss = self.NLLloss(output, target)

			break


			loss.backward()
			self.optimizer.step()


			if batch_idx % 100 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(self.train_loader.dataset),
					100. * batch_idx / len(self.train_loader), loss.data[0]))


	def test(self):
		self.model.eval()
		test_loss = 0
		correct = 0
		   
		for data, target in self.test_loader:

			if self.using_cuda:
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data.float(), volatile=True), Variable(target.long())
			output = self.model(data)
			test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
			pred = output.data.max(1)[1] # get the index of the max log-probability
			correct += pred.eq(target.data).cpu().sum()

		test_loss /= len(self.test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(self.test_loader.dataset),
			100. * correct / len(self.test_loader.dataset)))


	def saveModelToFile(self, path):
		torch.save(self.model.state_dict(), path)

	def loadModelFromFile(self, path):
		self.model.load_state_dict(torch.load(path))