import numpy as np
import cv2
import xml.etree.ElementTree
import os
from os import walk
import getGT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy
import struct
import cv2

z = Variable(torch.FloatTensor(4,3,2,2).zero_(), requires_grad=True)
for i in range(4):
	z.data[i][0] = 1 + i * 4
	z.data[i][1] = 2 + i * 4
	z.data[i][2] = 3 + i * 4

y = z.transpose(1,3).contiguous()

m = y.view(16, 3)

#print(m.transpose(0,1).contiguous().view(4,3,2,2))

print (m)


net = nn.Linear(3, 2)

n = net (m)
#print (n)
print(n.transpose(0,1).contiguous().view(4,2,2,2))