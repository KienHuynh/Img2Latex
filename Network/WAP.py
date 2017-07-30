from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy

def __init__(self):
		super(FCN, self).__init__()
		#self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
		self.conv1_1 = nn.Conv2d(1, 32, 3, stride=1,padding=1)
		self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
		self.conv1_4 = nn.Conv2d(32, 32, 3, stride=1,padding=1)

		self.pool_1 = nn.MaxPool2d(2, stride=1)

		self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
		self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv2_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)

		self.pool_2 = nn.MaxPool2d(2, stride=1)

		self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_3 = nn.Conv2d(64, 64, 3, stride=1,padding=1)
		self.conv3_4 = nn.Conv2d(64, 64, 3, stride=1,padding=1)

		self.pool_3 = nn.MaxPool2d(2, stride=1)

		self.conv4_1 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
		self.conv4_2 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_3 = nn.Conv2d(128, 128, 3, stride=1,padding=1)
		self.conv4_4 = nn.Conv2d(128, 128, 3, stride=1,padding=1)

		self.pool_4 = nn.MaxPool2d(2, stride=1)

		self.conv_temp = nn.Conv2d(128, 128, 3, stride=1,padding=1)



		self.fc1 = nn.Linear(73728, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):

		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.relu(self.conv1_3(x))
		x = F.relu(self.conv1_4(x))

		x = F.relu(self.pool_1(x))

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.relu(self.conv2_3(x))
		x = F.relu(self.conv2_4(x))

		x = F.relu(self.pool_2(x))

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = F.relu(self.conv3_4(x))

		x = F.relu(self.pool_3(x))

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = F.relu(self.conv4_4(x))

		x = F.relu(self.pool_4(x))

		x = x.view(-1, 73728)


		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		#return x

		return F.log_softmax(x)