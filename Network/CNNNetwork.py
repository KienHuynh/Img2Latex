from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import getGT
from torch.autograd import Variable
import numpy
numpy.set_printoptions(threshold=10000)

import WAP

import NetWorkConfig
import pdb
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
		self.learning_rate = 0.001
		self.momentum = 0.9
		self.lr_decay_base = 1/1.15
		
		self.model = WAP.WAP()

		#self.model = Net()
		#self.model = FCN()
		train_params = []
		for p in self.model.parameters(): 
			if (p.requires_grad):
				train_params.append(p)
				
		self.optimizer = optim.SGD(train_params, lr=self.learning_rate, momentum=self.momentum,
							 weight_decay = self.lr_decay_base)

		#self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
		self.NLLloss = nn.NLLLoss2d()
		self.NLLloss1 = nn.NLLLoss()
		self.criterion = nn.CrossEntropyLoss()
		
	def grad_clip(self, max_grad = 0.1):
		params = [p for p in list(self.model.parameters()) if p.requires_grad==True]
		for p in params:
			p_grad = p.grad 
			if (type(p_grad) == type(None)):
				#pdb.set_trace()
				#here = 1
				pass
			else:
				magnitude = torch.sqrt(torch.sum(p_grad**2)) 
				if (magnitude.data[0] > max_grad):
					p_grad.data = (max_grad*p_grad/magnitude.data[0]).data
					
					
					
	def try_print(self, print_flag = True):
		params = [p for p in list(self.model.parameters()) if p.requires_grad==True]
		for p in params:
			p_grad = p.grad 
			
			try:
				if print_flag:
					print ('exist')
					print (type(p_grad))
					print (p_grad.data.numpy().shape)
				else:
					print (p_grad.data.numpy())
					
			except:
				if print_flag:
					print ('non - exist')
					pass
			
			
	def setCudaState(self, state = True):
		self.using_cuda = state
		
	def setData(self, train, test):
		self.train_loader = train
		self.test_loader = test
		
	def train(self, epoch):
		self.model.train()
		###################decay _learning_rate################
#		lr = self.learning_rate
#		lr_decay_base = 1/1.15
#		epoch_base = 70
#		lr_decay = lr_decay_base ** max(epoch - epoch_base, 0)
#		lr = lr * lr_decay
		
		for batch_idx, (data, target) in enumerate(self.train_loader):
			if self.using_cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data.float()), Variable(target.long())
			self.optimizer.zero_grad()
			output = self.model(data)
			#print('output', output)
			
			if True:
				for b_id in range(NetWorkConfig.BATCH_SIZE):
					for s_id in range(50):
						if target.data[b_id, s_id] == getGT.word_to_id['$P']:
							target.data[b_id,s_id] = 0
							output.data[b_id,s_id, :] = 0
							output.data[b_id,s_id, 0] = 1
							
				target = target.view(NetWorkConfig.BATCH_SIZE * 50)
				output = output.view(NetWorkConfig.BATCH_SIZE * 50, NetWorkConfig.NUM_OF_TOKEN)
				
				loss = self.criterion(output, target)
				
			else :
				
				#######################3			

				#print (output)
				tar = Variable(torch.LongTensor(1).zero_(), requires_grad=False)
				tar.data[0] = 1
				#print (output)
				#tar.data[1] = 1
				loss = self.criterion(output, tar)

				#########################
			loss.backward()
			self.grad_clip()
			
#			self.try_print();
			
#			self.optimizer.step()
			self.optimizer.step()
			if batch_idx % 1 == 0:
#				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#						epoch, batch_idx * len(data), len(self.train_loader.dataset),
#						100. * batch_idx / len(self.train_loader), loss.data[0]))
				print(loss.data[0])
				if batch_idx > 1:
					pass
					#break
			#break
		
	
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

		