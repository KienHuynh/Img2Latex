from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

rnn = nn.GRUCell(128, 128)
input = Variable(torch.randn(2, 128))
x = Variable(torch.randn(2, 128))
x = rnn(input, x)
print (x)
	
	
	
#print (output)