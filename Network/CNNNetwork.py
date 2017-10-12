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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TestingNetwork:
	def __init__(self):
		print ('network init')
		self.all_loss = []
		self.all_lossA = []
		self.ite = 0
		################################################
		########## ATTRIBUTE INIT ######################
		################################################
		self.using_cuda = False  
		self.batch_size = 64
		self.learning_rate = 0.001
		#self.learning_rate = 0.0000131687243
		self.momentum = 0.9
		self.lr_decay_base = 0#1/1.15
		
		self.model = WAP.WAP()
		self.word_to_id, self.id_to_word = getGT.buildVocab('./parser/mathsymbolclass.txt')
		#self.model = Net()
		#self.model = FCN()
		conv_layers = [getattr(self.model, name) for name in dir(self.model) if type(getattr(self.model, name) )== type(self.model.conv1_3)]
		conv_params = []
		for c in conv_layers:
			for p in c.parameters():
				if (p.requires_grad):
					conv_params.append(p)
		other_layers = [module[1] for module in self.model.named_modules() if type(module[1]) != type(self.model.conv1_3)]
		other_layers = other_layers[1:]
		other_layers = [l for l in other_layers if hasattr(l,'parameters')]
		train_params= [] 
		for l in other_layers:
			for p in l.parameters():             
				train_params.append(p)
               				
		#self.optimizer = optim.SGD(train_params, lr=self.learning_rate, momentum=self.momentum,
		#					 weight_decay = self.lr_decay_base)
	                      
		self.optimizer = optim.Adam([
                        {'params': train_params},
                        {'params': conv_params, 'lr': 0.001}
                        #{'params': conv_params, 'lr': 1e-10}
                                        ], lr=self.learning_rate)
		#self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
		self.NLLloss = nn.NLLLoss2d()
		self.NLLloss1 = nn.NLLLoss()
		
		if NetWorkConfig.CURRENT_MACHINE == 0:
			self.criterion = nn.CrossEntropyLoss(ignore_index=1)
			self.criterion1 = nn.KLDivLoss()
		else:
			self.criterion = nn.CrossEntropyLoss()
		
	def grad_clip(self, max_grad = 1):
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
		
		self.model.setCuda(state)
		
	def setData(self, train, attendGT, test):
		
		self.train_loader = train
		self.test_loader = test
		self.attendGT = attendGT
#		pdb.set_trace()
	
	def train(self, epoch):
		self.model.train()
		###################decay _learning_rate################
#		lr = self.learning_rate
#		lr_decay_base = 1/1.15
#		epoch_base = 70
#		lr_decay = lr_decay_base ** max(epoch - epoch_base, 0)
#		lr = lr * lr_decay

		for batch_idx, (data, target) in enumerate(self.train_loader):

			self.model.setGroundTruth(target.numpy())

			if self.using_cuda:
				data, target = data.cuda(), target.cuda()
				
			if (self.ite%3000 == 2999):
				self.learning_rate = self.learning_rate/1.5
				print("Current learning rate: %.8f" % self.learning_rate)
				self.optimizer.param_groups[0]['lr'] = self.learning_rate
				self.optimizer.param_groups[1]['lr'] /= 1.5
				print(self.optimizer.param_groups[0]['lr'])

			data, target = Variable(data.float()), Variable(target.long())

			self.optimizer.zero_grad()
			output, attendOut = self.model(data)
			attendOut = attendOut[:,1:-1,:,:]
			if self.using_cuda:
				attendGT = Variable(torch.FloatTensor(self.attendGT)).cuda()
			else:
				attendGT = Variable(torch.FloatTensor(self.attendGT))
			ignore_block = numpy.zeros((1,89,16,32))
			ignore_vecs = numpy.zeros((1, 89, 1, 1))
			for batch_index in range(NetWorkConfig.BATCH_SIZE):
				target_vec = target[batch_index].cpu().data.numpy()
				
				ignore_vec= numpy.copy(target_vec)
				ignore_vec[ignore_vec==1]=-1
				ignore_vec[ignore_vec!=-1]=1
				ignore_vec[ignore_vec==-1]=0
				ignore_vec = ignore_vec[1:]
				ignore_vec = numpy.reshape(ignore_vec, (1,89,1,1))
				ignore_vecs = numpy.concatenate((ignore_vecs, ignore_vec), axis=0)
				ignore_map = numpy.tile(ignore_vec, (1,1,16,32))
				ignore_block = numpy.concatenate((ignore_block, ignore_map),0)
			ignore_block = ignore_block[1:,:,:,:]
				
			if self.using_cuda:
				attendGT = Variable(torch.FloatTensor(self.attendGT)).cuda()
				ignore_block =Variable(torch.FloatTensor(ignore_block)).cuda()
			else:
				attendGT = Variable(torch.FloatTensor(self.attendGT))
				ignore_block =Variable(torch.FloatTensor(ignore_block))
				
			
#			target_vec = target.cpu().data.numpy()
#			pdb.set_trace()
			#print('output', output)
#			pdb.set_trace()
			if True:
				#for b_id in range(NetWorkConfig.BATCH_SIZE):
				#	for s_id in range(50):
				#		if target.data[b_id, s_id] == getGT.word_to_id['$P']:
				#			target.data[b_id,s_id] = 0
				#			output.data[b_id,s_id, :] = 0
				#			output.data[b_id,s_id, 0] = 1
				#pdb.set_trace()			
				#target = target[:,0:49]
				target.contiguous()
				#output = output[:,0:50]
				output.contiguous()

				target_vec = target.cpu().data.numpy()
				output_vec = numpy.argmax(output.cpu().data.numpy(), axis=2)
				target = target.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN)
				output = output.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)
				target_vec = target_vec[0][0:30]
				output_vec = output_vec[0][0:30]
				target_str = []
				output_str = []
				for idx in range(0,30):
					target_str.append(self.id_to_word[target_vec[idx]])
					output_str.append(self.id_to_word[output_vec[idx]])

				print('target', ' '.join(target_str))
				print('output', ' '.join(output_str))
			        

				#-------------------------------------------------

				loss = self.criterion(output, target)
#				pdb.set_trace()
				lossA = 10*torch.sum(attendGT * (torch.log(attendGT)-torch.log(attendOut))*ignore_block)/numpy.sum(ignore_vecs)
                                
				loss = loss + lossA 
			else :
				pass

			loss.backward()
			self.grad_clip()
                        #pdb.set_trace()
			if (epoch % 20 == 0):
				pass
#				pdb.set_trace()
#			self.try_print();
                        #pdb.set_trace()
			self.model.Coverage_MLP_From_H
			self.optimizer.step() 
			#print(list(self.model.conv3_1.parameters())[0].norm().cpu().data.numpy())
			#print(list(self.model.Coverage_MLP_From_A.parameters())[0].norm().cpu().data.numpy())

			self.ite += 1
			self.all_loss.append(loss.cpu().data.numpy())
			self.all_lossA.append(lossA.cpu().data.numpy())
			plt.ion()

			if batch_idx % 1 == 0:
#				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#						epoch, batch_idx * len(data), len(self.train_loader.dataset),
#						100. * batch_idx / len(self.train_loader), loss.data[0]))
				print('[E %d, I %d]: %.5f' % (epoch,self.ite, loss.data[0]))
				print('lossA',lossA)
				if batch_idx > 1:
					pass
					if self.ite % 500 == 0:
						plt.clf()
				try:
					plt.plot(self.all_loss)
					plt.savefig('figures/total_loss.png')
					plt.clf()
					plt.plot(self.all_lossA)
					plt.savefig('figures/total_lossA.png')
					plt.clf()
				except:
					pass
				#plt.draw()
			#break
		
	
	def test(self, evaluation_method = 0, batch_size  = 1):
		self.model.eval()
		#self.model.train()
		
		for batch_idx, (data, target) in enumerate(self.train_loader):


			self.model.setGroundTruth(target.numpy())
			if self.using_cuda:
				print('using cuda', self.using_cuda)
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data.float()), Variable(target.long())
			
			output = self.model(data)
			#print('output', output)
			

			target.contiguous()
			output.contiguous()

			target = target.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN)
			output = output.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)

			#print ('------PTP DEBUG------------------');
			#print (target.view(1, NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN))
			#print(output.max(1)[1].view(1, NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN))				        

				#-------------------------------------------------
			self.ite += 1
			

			sum_loss = 0

			for i in range(batch_size):
				sum_loss = sum_loss + self.testFunction(output.max(1)[1].cpu().data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], target.data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], evaluation_method)

			return sum_loss / float(batch_size)




	def testAll(self, batch_size  = 1):
		self.model.eval()
		self.model.train()
		
		for batch_idx, (data, target) in enumerate(self.train_loader):


			self.model.setGroundTruth(target.numpy())
			if self.using_cuda:
				print('using cuda', self.using_cuda)
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data.float()), Variable(target.long())
			
			output = self.model(data)
			#print('output', output)
			

			target.contiguous()
			output.contiguous()


			target_vec = target.cpu().data.numpy()
			output_vec = numpy.argmax(output.cpu().data.numpy(), axis=2)
			target = target.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN)
			output = output.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)
			target_vec = target_vec[0][0:40]
			output_vec = output_vec[0][0:40]
			target_str = []
			output_str = []
			for idx in range(0,20):
				target_str.append(self.id_to_word[target_vec[idx]])
				output_str.append(self.id_to_word[output_vec[idx]])
			print('target', ' '.join(target_str))
			print('output', ' '.join(output_str))


			target = target.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN)
			output = output.view(NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)

			#print ('------PTP DEBUG------------------');
			#print (target.view(1, NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN))
			#print(output.max(1)[1].view(1, NetWorkConfig.BATCH_SIZE * NetWorkConfig.MAX_TOKEN_LEN))				        

				#-------------------------------------------------
			self.ite += 1
			

			sum_loss_w = 0
			sum_loss_l = 0
			sum_loss_e = 0

			for i in range(batch_size):
				sum_loss_w = sum_loss_w + self.testFunction(output.max(1)[1].cpu().data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], target.data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], 0)

				sum_loss_l = sum_loss_l + self.testFunction(output.max(1)[1].cpu().data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], target.data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], 1)

				sum_loss_e = sum_loss_e + self.testFunction(output.max(1)[1].cpu().data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], target.data.numpy()[i * NetWorkConfig.MAX_TOKEN_LEN : (i + 1) * NetWorkConfig.MAX_TOKEN_LEN], 2)

			return sum_loss_w / float(batch_size), sum_loss_l / float(batch_size), sum_loss_e / float(batch_size)

		
	# stragety:
	# 0: expression
	# 1: word-distance
	# 2: Euclid
	def testFunction(self, predict, expect, stragety):

		

		predict = predict.flatten()

		if stragety == 0:
			for i in range(NetWorkConfig.MAX_TOKEN_LEN):
				
				

				if predict[i] != expect[i]:
					if (predict[i] == self.word_to_id['$P'] or predict[i] == self.word_to_id['</s>']) and expect[i] == self.word_to_id['$P']:
						return 0
					return 1
				elif expect[i] == self.word_to_id['</s>']:
					return 0			
			return 1
		if stragety == 1:
			## get len
			word_len = NetWorkConfig.MAX_TOKEN_LEN
			for i in range(NetWorkConfig.MAX_TOKEN_LEN):

				if (predict[i] == self.word_to_id['$P'] or predict[i] == self.word_to_id['</s>']) and expect[i] == self.word_to_id['$P']:
					word_len = i
					break
				
			#print(self.LevenshteinDistance(expect[0: word_len], predict[0: word_len]))
			return self.LevenshteinDistance(expect[0: word_len], predict[0: word_len])

		if stragety == 2:
			word_len = NetWorkConfig.MAX_TOKEN_LEN

			incorrect_count = 0

			for i in range(NetWorkConfig.MAX_TOKEN_LEN):

				if (predict[i] == self.word_to_id['$P'] or predict[i] == self.word_to_id['</s>']) and expect[i] == self.word_to_id['$P']:
					word_len = i
					break
				else:
					if predict[i] != expect[i]:
						incorrect_count = incorrect_count + 1

			return incorrect_count / float(word_len - 1)



		


	def LevenshteinDistance(self, s, t):

		m = len(s)
		n = len(t)

		d = numpy.zeros((m + 1, n + 1))

		for i in range(m + 1):
			d[i, 0] = i

		for j in range(n + 1):
			d[0, j] = j

		

		for j in range(1, n + 1):
			for i in range(1, m + 1):
				if s[i - 1] == t[j - 1]:
					substitutionCost = 0
				else:
					substitutionCost = 1
				d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + substitutionCost)

		return d[m, n] / max(m , n)

	def printTestResult(self,  target, predict):
		target = target.cpu().data.numpy()
		predict = predict.max(1)[1].cpu().data.numpy()

		#self.getSampleResult(target, predict, self.ite)

		for j in range(NetWorkConfig.BATCH_SIZE):
			print('')
			print ('Batch index: ' + str(j))
			for i in range(NetWorkConfig.MAX_TOKEN_LEN ):
				if ((target[i + j * NetWorkConfig.MAX_TOKEN_LEN] == 1) and (target[i + j * NetWorkConfig.MAX_TOKEN_LEN + 1] == 1)):
					break
				print ('%3s|' % target[i + j * NetWorkConfig.MAX_TOKEN_LEN], end='')
			print ('')
			for i in range(NetWorkConfig.MAX_TOKEN_LEN):
				if ((predict[i + j * NetWorkConfig.MAX_TOKEN_LEN] == 19) and (predict[i + j * NetWorkConfig.MAX_TOKEN_LEN + 1] == 19)):
					break
				print ('%3s|' % predict[i + j * NetWorkConfig.MAX_TOKEN_LEN], end='')

	def getSampleResult(self, target, predict, batch_id):
		numpy.save('./PTPsSample/target'  + str(batch_id), target)
		numpy.save('./PTPsSample/predict' + str(batch_id), predict)


	def saveModelToFile(self, path):
		torch.save(self.model.state_dict(), path)
		try:
			torch.save(self.model, path + 'abc')
		except:
			pass 
	
	def loadModelFromFile(self, path):
		self.model.load_state_dict(torch.load(path))
		#self.model = torch.load(path)

	def saveLearningRate(self):

		try:
			f = open('lr.txt', w)
			f.write(str(self.learning_rate))
			f.close()

		except:
			print ('The code is malfunction! anyway, current lr is:')
			print (self.learning_rate)
