#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:30:37 2017

@author: ngocbui
"""

import sys
sys.path.insert(0, './../Network')

import numpy as np
import xml.etree.ElementTree as r
import torch
from torch.autograd import Variable
import os
import re
import collections
import NetWorkConfig

def readSymbolfile(path):
    assert(os.path.exists(path))
    with open(path, 'r') as f:
        return f.read().replace("\n", " ").split()
    
def buildVocab(path):
    
    data = readSymbolfile(path)
    counter = collections.Counter(data)
#    print(counter)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
#    print(count_pairs)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())
#    print(word_to_id)
#    print(id_to_word)
#    train = _file_to_word_ids(truth, word_to_id)
#    print(train)
    return word_to_id, id_to_word

word_to_id, id_to_word = buildVocab('./../tool/mathsymbolclass.txt')
