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



v1 = Variable(torch.FloatTensor(2,3,3).zero_(), requires_grad=True)
v2 = Variable(torch.FloatTensor(1,3,3).zero_(), requires_grad=True)

v1.data[0][0][0] = 1
v1.data[0][0][1] = 2
v1.data[0][0][2] = 3
v1.data[0][1][0] = 4
v1.data[0][1][1] = 5
v1.data[0][1][2] = 6
v1.data[0][2][0] = 7
v1.data[0][2][1] = 8
v1.data[0][2][2] = 9

v1.data[1][0][0] = 12
v1.data[1][0][1] = 2
v1.data[1][0][2] = 3
v1.data[1][1][0] = 4
v1.data[1][1][1] = 5
v1.data[1][1][2] = 6
v1.data[1][2][0] = 7
v1.data[1][2][1] = 8
v1.data[1][2][2] = 9

v2.data[0][0][0] = 1
v2.data[0][0][1] = 2
v2.data[0][0][2] = 3
v2.data[0][1][0] = 4
v2.data[0][1][1] = 5
v2.data[0][1][2] = 6
v2.data[0][2][0] = 7
v2.data[0][2][1] = 8
v2.data[0][2][2] = 9


#print (v1)
print (v2)

print (v2.repeat(2))

quit()

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
oo = n.transpose(0,1).contiguous().view(4,2,2,2)

print (oo)
oo = oo + 100
print (oo)