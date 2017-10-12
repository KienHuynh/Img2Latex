#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:11:23 2017

@author: ngocbui
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
import numpy
import struct
import pdb
import scipy.misc
from scipy.misc import imresize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xml.etree.ElementTree
import os
from os import walk
import re
import collections
import NetWorkConfig

def readSymbolfile(path):
#	pdb.set_trace()
	assert(os.path.exists(path))
	with open(path, 'r') as f:
		return f.read().replace("\n", " ").split()
	
def buildVocab(path):
	
	data = readSymbolfile(path)
	counter = collections.Counter(data)
	#print(counter)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	#print(count_pairs)
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = dict((v, k) for k, v in word_to_id.items())
	#print(len(word_to_id))
	#print(id_to_word)
	#train = _file_to_word_ids(truth, word_to_id)
	#print(train)
	return word_to_id, id_to_word

def parseOfficialV_4(input_path, scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
	
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path
#	pdb.set_trace()
	try:
		try:
			root = xml.etree.ElementTree.parse(input_path).getroot()
#			pdb.set_trace()
		except:
			print ('error parsing',input_path)
			pdb.set_trace()
			return
		tag_header_len = len(root.tag) - 3
		vertex_arr = []
		min_x = 999999
		min_y = 999999
		max_x = 0
		max_y = 0
		for child in root:
			tag = child.tag[tag_header_len:]
			################################
			####### GET VERTICES ###########
			################################
			if tag == 'trace':
				temp_arr = []
				processing_text = child.text
				try:
					test_str = processing_text[:processing_text.index(',')]
					test_str = test_str.strip()
					test_str = test_str.split(' ')
					vertexlen = len(test_str)
				except:
					vertexlen = 2
				processing_text = processing_text.replace(',', '')
				processing_text = processing_text.replace('\n', '')
				raw_vertex_list = processing_text.split(' ')
				for i in range(int(len(raw_vertex_list) / vertexlen)):
					x = float(raw_vertex_list[vertexlen * i])
					y = float(raw_vertex_list[vertexlen * i + 1])
					if x > max_x:
						max_x = x
					if y > max_y:
						max_y = y
					if x < min_x:
						min_x = x
					if y < min_y:
						min_y = y
					temp_arr.append ((x, y))
				vertex_arr.append(temp_arr)
				
		#################################
		##### GENERATE ##################
		#################################
		output = np.zeros((target_height, target_width))
		width = max_x - min_x
		heigh = max_y - min_y
		evaluate_width = width * scale_factor
		evaluate_heigh = heigh * scale_factor
		if evaluate_width > (target_width - padding):
			scale_factor = scale_factor * (target_width - padding) / float(evaluate_width)
		
		if evaluate_heigh > (target_height - padding):
			scale_factor = scale_factor * (target_height - padding) / float(evaluate_heigh)
		
		expr_img = np.zeros((int(heigh * scale_factor) + 1 , int(width * scale_factor) + 1 ))
		
		for stroke in vertex_arr:
			
			#print(len(vertex_arr))

			temp_vertex_arr = []
			for vertex in stroke:
				temp_vertex_arr.append((int((vertex[0] - min_x) * scale_factor ), int((vertex[1] - min_y) * scale_factor)))
			
			if (len(stroke) < 2):
				cv2.circle(expr_img, temp_vertex_arr[0], 1, 255, 2)
				continue
				
			
			for i in range (len(stroke) - 1):
				cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)
		#################################
		##### PADDING ###################
		#################################
		y_offset = int((target_height - expr_img.shape[0]) / 2)
		x_offset = int((target_width - expr_img.shape[1]) / 2)
		
		output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img
		
		#cv2.imwrite(output_path, output)
		#print('dacoanh')
		eight_imgs = []
		for theta in np.arange(0,2*np.pi, np.pi / 4):
			
			kernel = cv2.getGaborKernel((30,30), 3.0, theta, 5.0, 0.5, 0, cv2.CV_32F)
			filtered_img = cv2.filter2D(output, cv2.CV_8UC3, kernel)
			eight_imgs.append(filtered_img)
		eight_imgs.append(output) 
		
		###########################################################
		################ Attention ################################
		##########################################################		

		attention_list = []

#		pdb.set_trace()
		word_to_id, id_to_word = buildVocab('./parser/mathsymbolclass.txt')
#		word_to_id, id_to_word = sharedFunction.buildVocab('./mathsymbolclass.txt')
		for child in root:
			tag = child.tag[tag_header_len:]
			if tag == 'traceGroup':
				traceGroupRoot = child
				for TGchild in traceGroupRoot:

					

					trace_list = []

					sum_of_pts_x = 0
					sum_of_pts_y = 0
					pts_count = 0
					minx = 99999
					miny = 99999
					maxx = 0
					maxy = 0
					t_minx = 0
					t_miny =0
					t_maxx=0
					t_maxy=0

					for TGGrandChild in TGchild:
						childtag = TGGrandChild.tag[tag_header_len:]
						#print(childtag)
#						pdb.set_trace()
						if childtag == 'annotationXML':
							href = TGGrandChild.get('href')

						if childtag == 'annotation':
							anno = TGGrandChild.text

						if childtag == 'traceView':
							trace_id = int(TGGrandChild.get('traceDataRef'))

							for i in range(len(vertex_arr[trace_id])):

#								print ((vertex_arr[trace_id][i][0] - min_x)*scale_factor)
#								print(vertex_arr[trace_id][i][0])
#								print(vertex_arr[trace_id][i][1])
								sum_of_pts_x += vertex_arr[trace_id][i][0] - min_x
								sum_of_pts_y += vertex_arr[trace_id][i][1] - min_y
							
							pts_count += len(vertex_arr[trace_id])
#							print(vertex_arr[trace_id][:][0])
							list_x = [ele[0] for ele in vertex_arr[trace_id][:]]
							list_y = [ele[1] for ele in vertex_arr[trace_id][:]]
							t_minx = (min(list_x) - min_x)*scale_factor + x_offset
							t_miny = (min(list_y) - min_y)*scale_factor + y_offset
							t_maxx = (max(list_x) - min_x)*scale_factor + x_offset
							t_maxy = (max(list_y) - min_y)*scale_factor + y_offset
							
#							pdb.set_trace()
					if t_minx < minx:
						minx = t_minx
					if t_miny < miny:
						miny = t_miny
					if t_maxx > maxx:
						maxx = t_maxx
					if t_maxy > maxy:
						maxy = t_maxy
					if pts_count > 0:
						av_x = sum_of_pts_x * scale_factor / pts_count + x_offset
						av_y = sum_of_pts_y * scale_factor / pts_count + y_offset
						
#						pdb.set_trace()
						retdata = (anno, href, int(av_x), int(av_y), minx, miny, maxx, maxy)
						
						attention_list.append(retdata)
#		print(attention_list)

#		attentionGT = createAttentionGT(attention_list, input_path, './parser/mathsymbolclass.txt')
#		attentionGT = createAttentionGT(attention_list, input_path, './mathsymbolclass.txt')

		
		return eight_imgs, attention_list
	except NotImplementedError as e:
		print (e)
		print ('error parse ' + input_path)
		return np.array([])
def checkFile(input_path):
	try:
		root = xml.etree.ElementTree.parse(input_path).getroot()
#			pdb.set_trace()
	except:
		print ('error parsing',input_path)
		pdb.set_trace()
		return
	tag_header_len = len(root.tag) - 3
	for child in root:
		tag = child.tag[tag_header_len:]
		if tag == 'traceGroup':
			traceGroupRoot = child
			for TGchild in traceGroupRoot:
				num = 0
#				print(TGchild)
				for TGGrandChild in TGchild:
					childtag = TGGrandChild.tag[tag_header_len:]
#					print(childtag)
					if childtag == 'annotationXML':
						num+=1
				
				if num==0 and TGchild.tag[tag_header_len:]!= 'annotation':
#					pdb.set_trace()
					return False
	return True
#print(checkFile('./../data/TrainINKML/HAMEX/formulaire004-equation041.inkml'))
def parseGT1(root, text, ignoreElems):
	index = getIndex(root)
	if root.tag[index:] in ignoreElems:
		return
	if len(root) == 0:
		temp = modifiedText(root.text)
		text.append(temp)
		tempID = getAttribVal(root)
#		textID.append(getAttribVal(root))
		return (temp,tempID)
	else:
	#	print(root.tag[length+6:])
	#	print('tttag',root.tag)
		if root.tag[index:] == 'msqrt':
			sqrt_scope = []
			sqrt_scope.append(('\\sqrt',getAttribVal(root)))
			sqrt_scope.append(('{','{'))
			text.append('\\sqrt')
			text.append('{')
			for child in root:
				res = parseGT1(child, text, ignoreElems)
				sqrt_scope.append(res)
			text.append('}')
			sqrt_scope.append(('}','}'))
			return sqrt_scope
		elif root.tag[index:] == 'mroot':
			n = 1
			text.append('\\sqrt')
			scope_list = []
			scope_list.append(('\\sqrt',getAttribVal(root)))
			for child in root:
				if n == 2:
					scope_list.append(('{','{'))
					text.append('{')
					res = parseGT1(child, text, ignoreElems)
					scope_list.append(res)
					scope_list.append(('}','}'))
					text.append('}')
				else:
					text.append('[')
					scope_list.append(('[','['))
					res = parseGT1(child, text, ignoreElems)
					scope_list.append(res)
					scope_list.append((']',']'))
					text.append(']')
				n += 1
			return scope_list
		elif root.tag[index:] == 'mfrac':
			scope_list = []
			scope_list.append(('\\frac',getAttribVal(root)))
			text.append('\\frac')
			for child in root:
				text.append('{')
				scope_list.append(('{','{'))
				res = parseGT1(child, text, ignoreElems)
				scope_list.append(res)
				scope_list.append(('}','}'))
				text.append('}')  
			return scope_list
		elif root.tag[index:] == 'msub' or root.tag[index:] == 'munder':
			n = 1
			scope_list = []
			for child in root:
				if n == 2:
					scope_list.append(('_','_'))
					text.append('_')
					if child.tag[index:] == 'mrow':
						text.append('{')
						scope_list.append(('{','{'))
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
						scope_list.append(('}','}'))
						text.append('}')
					else:
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
				else:
					res = parseGT1(child, text, ignoreElems)
					scope_list.append(res)
				n = n + 1
			return scope_list
		elif root.tag[index:] == 'msup' or root.tag[index:] == 'mover':
			n = 1
			scope_list = []
			for child in root:
				if n == 2:
					scope_list.append(('^','^'))
					text.append('^')
					if child.tag[index:] == 'mrow':
						text.append('{')
						scope_list.append(('{','{'))
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
						scope_list.append(('}','}'))
						text.append('}')
					else:
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
				else:
					res = parseGT1(child, text, ignoreElems)
					scope_list.append(res)
				n = n + 1
			return scope_list
		elif root.tag[index:] == 'msubsup' or root.tag[index:] == 'munderover':
			n = 1
			scope_list = []
			for child in root:
				if n == 2:
					text.append('_')
					scope_list.append(('_','_'))
					if child.tag[index:] == 'mrow':
						text.append('{')
						scope_list.append(('{','{'))
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
						scope_list.append(('}','}'))
						text.append('}')
					else:
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
				elif n == 3:
					text.append('^')
					scope_list.append(('^','^'))
					if child.tag[index:] == 'mrow':
						text.append('{')
						scope_list.append(('{','{'))
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
						scope_list.append(('}','}'))
						text.append('}')
					else:
						res = parseGT1(child, text, ignoreElems)
						scope_list.append(res)
				else:
					res = parseGT1(child, text, ignoreElems)
					scope_list.append(res)
				n = n + 1
			return scope_list
	#	elif root.tag[index:] == 'munder':
			
		else:
			index = getIndex(root)
#			pdb.set_trace()
			if root.tag[index:] == 'math':
#				pdb.set_trace()
				express_tree = []
				for child in root:
					res = parseGT1(child, text, ignoreElems)
#					pdb.set_trace()
					express_tree.append(res)
#				pdb.set_trace()
				return express_tree
			else:
#				pdb.set_trace()
				scope_list = []
				for child in root:
					res = parseGT1(child, text, ignoreElems)
#					pdb.set_trace()
					if res != None:
						scope_list.append(res)
				return scope_list
			

def readSymbolfile(path):
#	pdb.set_trace()
	assert(os.path.exists(path))
	with open(path, 'r') as f:
		return f.read().replace("\n", " ").split()
	
def buildVocab(path):
	
	data = readSymbolfile(path)
	counter = collections.Counter(data)
	#print(counter)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	#print(count_pairs)
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = dict((v, k) for k, v in word_to_id.items())
	#print(len(word_to_id))
	#print(id_to_word)
	#train = _file_to_word_ids(truth, word_to_id)
	#print(train)
	return word_to_id, id_to_word


def replaceW2ID(data, word_to_id):
	#print('data', data)
	#print('{',word_to_id['<s>'])
	#data = readSymbolfile(path)
	return [word_to_id[word] for word in data if word in word_to_id]

def touchGT(path):
	assert(os.path.exists(path))
	root = r.parse(path).getroot()
	#print('parse', r.parse(path))
	#print(root.tag)
	#print(root.getchildren())
	tag_header_len = len(root.tag)-3
	
	for child in root:
		tag = child.tag[tag_header_len:]
		#print(tag)
		if tag == 'annotation' and child.attrib['type'] == 'truth':
			text = child.text
			#print(text)
			text = text.replace('$','')
			#print(text)
	#		print(text.split())
	#		text = text.split()
	return text


def getRoot(path):
	#print('path', path)
	assert(os.path.exists(path))
	root = xml.etree.ElementTree.parse(path).getroot()
	return root


def modifiedText(text):
	standard = ['phi','pi','theta','alpha','beta','gamma','infty','sigma','Delta',
				'lamda','mu','pm','sin','cos','neq','leq','gt','sqrt','div','times',
				'sum','log','tan','ldots','geq','rightarrow','lim','int','exists',
				'forall','in','prime','lt','ne','cdot','cdots','{','}']
	
	if text == '<':
		text = 'lt'
	elif text =='>':
		text = 'gt'
	elif text == 'im':
		text = 'lim'
	elif text == '.':
		text = 'cdot'
	elif text == 'ctdot':
		text = 'cdots'
	elif text == '\\ctdot':
		text = 'cdots'
		
	if text in standard:
		standtext = '\\'+text
	else:
		standtext = text
	return standtext


def getIndex(root):
	index = root.tag.index('}') + 1
	return index

	
#def ParseGTFromfile(root, text, ignoreElems):
	#text.append('$B')
	#parseGT(root, text, ignoreElems)
	#text.append('$E')
	#


def centerofSymbol(root,textID, attention_list):
	tempID = getAttribVal(root)
	textID.append(tempID)
	symbol = [elem for elem in attention_list if tempID in elem]
#	res = (tempID, symbol[2], symbol[3], symbol[4], symbol[5], symbol[6], symbol[7])
	return symbol


#def parseGT1(root, text,textID, attention_list, xy_list, ignoreElems):
#	index = getIndex(root)
#	if root.tag[index:] in ignoreElems:
#		return
#	if len(root) == 0:
#		temp = modifiedText(root.text)
#		text.append(temp)
#		res = centerofSymbol(root,textID, attention_list)
#		xy_list.append(res)
#		return res
#	else:
#	#	print(root.tag[length+6:])
#	#	print('tttag',root.tag)
#		if root.tag[index:] == 'msqrt':
#			
#			text.append('\\sqrt')
#			res1 = centerofSymbol(root,textID, attention_list)
#			xy_list.append(res1)
#			text.append('{')
#			n == 1
#			for child in root:
#				res2 = parseGT(child, text, ignoreElems)
#				if n == 1:
#					x = (res1[2]+res2[2])/32
#					y = (res1[3]+res2[3])/32
#					xy_list.append(('{',x,y, 0,0,5,5))
#				n+=1
#			text.append('}')
#			pre_symbol = xy_list[-1]
#			xy_append(('}',pre_symbol[2] + 1, pre_symbol[3] +1, 0,0,5,5))
#		elif root.tag[index:] == 'mroot':
#			n = 1
#			text.append('\\sqrt')
#			res1 = centerofSymbol(root, attention_list)
#			xy_list.append(res1)
#			for child in root:
#				if n == 2:
#					text.append('{')
#					i = 1
#					for subchild in child:
#						res2 = parseGT(subchild, text, ignoreElems)
#						if i == 1:
#							x = (res1[2]+res2[2])/32
#							y = (res1[3]+res2[3])/32
#							xy_list.append(('{',x,y,0,0, 5,5))
#						i+=1
#					text.append('}')
#					pre_symbol = xy_list[-1]
#					xy_append(('}',pre_symbol[2] + 1, pre_symbol[3] +1,0,0 ,5,5))
#				else:
#					text.append('[')
#					res2 = parseGT(child, text, ignoreElems)
#					x = (res1[2]+res2[2])/32
#					y = (res1[3]+res2[3])/32
#					xy_list.append(('[',x,y, 5,5))
#					text.append(']')
#					pre_symbol = xy_list[-1]
#					xy_append((']',pre_symbol[2] + 1, pre_symbol[3] +1,0,0, 5,5))
#				n += 1
#			
#		elif root.tag[index:] == 'mfrac':
#			text.append('\\frac')
#			res1 = centerofSymbol(root, attention_list)
#			xy_list.append(res1)
#			for child in root:
#				text.append('{')
#				i = 1
#				for subchild in child:
#					res2 = parseGT(child, text, ignoreElems)
#					if i == 1:
#						x = res2[2]-int(round((res1[6]-res1[4])/32))
#						y = (res1[3]+res2[3])/32
#						xy_list.append(('{',x,y,0,0, 5,5))
#					i+=1
#				text.append('}')  
#				pre_symbol = xy_list[-1]
#				xy_append(('}',pre_symbol[2] + 1, pre_symbol[3] +1,0,0, 5,5))
#		elif root.tag[index:] == 'msub' or root.tag[index:] == 'munder':
#			n = 1
#			for child in root:
#				if n == 2:
#					text.append('_')
#					pre_symbol = xy_list[-1]
#					
#					if child.tag[index:] == 'mrow':
#						text.append('{')
#						parseGT(child, text, ignoreElems)
#						text.append('}')
#					else:
#						parseGT(child, text, ignoreElems)
#				else:
#					parseGT(child, text, ignoreElems)
#				n = n + 1
#		elif root.tag[index:] == 'msup' or root.tag[index:] == 'mover':
#			n = 1
#			for child in root:
#				if n == 2:
#					text.append('^')
#					if child.tag[index:] == 'mrow':
#						text.append('{')
#						parseGT(child, text, ignoreElems)
#						text.append('}')
#					else:
#						parseGT(child, text, ignoreElems)
#				else:
#					parseGT(child, text, ignoreElems)
#				n = n + 1
#		elif root.tag[index:] == 'msubsup' or root.tag[index:] == 'munderover':
#			n = 1
#			for child in root:
#				if n == 2:
#					text.append('_')
#					if child.tag[index:] == 'mrow':
#						text.append('{')
#						parseGT(child, text, ignoreElems)
#						text.append('}')
#					else:
#						parseGT(child, text, ignoreElems)
#				elif n == 3:
#					text.append('^')
#					if child.tag[index:] == 'mrow':
#						text.append('{')
#						parseGT(child, text, ignoreElems)
#						text.append('}')
#					else:
#						parseGT(child, text, ignoreElems)
#				else:
#					parseGT(child, text, ignoreElems)
#				n = n + 1
#	#	elif root.tag[index:] == 'munder':
#			
#		else:
#			n = 1
#			for child in root:
#				res = parseGT(child, text, ignoreElems)
#				if n== 1:
#					pre_symbol = xy_list[-1]
#					if pre_symbol[0] == '\\sqrt':
#						x = (pre_symbol[2]+res[2])/32
#						y = (pre_symbol[3]+res[3])/32
#						xy_list.append(('{',x,y, 0,0,5,5))
#				pre_symbol = 
def getAttribVal(root):
	dict = root.attrib
	field, value = dict.items()[0]
	return value

def parseGTid(root, text, ignoreElems):
	index = getIndex(root)
	if root.tag[index:] in ignoreElems:
		return
	if len(root) == 0:
		text.append(getAttribVal(root))
		return
	else:
	#	print(root.tag[length+6:])
	#	print('tttag',root.tag)
		if root.tag[index:] == 'msqrt':
			text.append(getAttribVal(root))
			text.append('{')
			for child in root:
				parseGTid(child, text, ignoreElems)
			text.append('}')
		elif root.tag[index:] == 'mroot':
			n = 1
			text.append(getAttribVal(root))
			for child in root:
				if n == 2:
					text.append('{')
					parseGTid(child, text, ignoreElems)
					text.append('}')
				else:
					text.append('[')
					parseGTid(child, text, ignoreElems)
					text.append(']')
				n += 1
			
		elif root.tag[index:] == 'mfrac':
			text.append(getAttribVal(root))
			for child in root:
				text.append('{')
				parseGTid(child, text, ignoreElems)
				text.append('}')  
		elif root.tag[index:] == 'msub' or root.tag[index:] == 'munder':
			n = 1
			for child in root:
				if n == 2:
					text.append('_')
					if child.tag[index:] == 'mrow':
						text.append('{')
						parseGTid(child, text, ignoreElems)
						text.append('}')
					else:
						parseGTid(child, text, ignoreElems)
				else:
					parseGTid(child, text, ignoreElems)
				n = n + 1
		elif root.tag[index:] == 'msup' or root.tag[index:] == 'mover':
			n = 1
			for child in root:
				if n == 2:
					text.append('^')
					if child.tag[index:] == 'mrow':
						text.append('{')
						parseGTid(child, text, ignoreElems)
						text.append('}')
					else:
						parseGTid(child, text, ignoreElems)
				else:
					parseGTid(child, text, ignoreElems)
				n = n + 1
		elif root.tag[index:] == 'msubsup' or root.tag[index:] == 'munderover':
			n = 1
			for child in root:
				if n == 2:
					text.append('_')
					if child.tag[index:] == 'mrow':
						text.append('{')
						parseGTid(child, text, ignoreElems)
						text.append('}')
					else:
						parseGTid(child, text, ignoreElems)
				elif n == 3:
					text.append('^')
					if child.tag[index:] == 'mrow':
						text.append('{')
						parseGTid(child, text, ignoreElems)
						text.append('}')
					else:
						parseGTid(child, text, ignoreElems)
				else:
					parseGTid(child, text, ignoreElems)
				n = n + 1
	#	elif root.tag[index:] == 'munder':
			
		else:
			for child in root:
				parseGTid(child, text, ignoreElems)
#				
def makeGTidVec(path_to_ink, path_to_symbol, scale_factor,padding):
	word_to_id, id_to_word = buildVocab(path_to_symbol)
	#print(id_to_word)
	#chuan hoa text de tach ra duoc tung symbol va luu thanh mang trong data
	#TODO
	#data = ['\\forall', 'g', '\\in', 'G'] 
	#print(touchGT(path_to_ink))
	root = getRoot(path_to_ink)
	im, attention_list = parseOfficialV_4(path_to_ink,scale_factor, padding=padding)
#	print('attend',attention_list)
	ignoreElems = ['traceFormat','annotation','trace','traceGroup']
	text = ['<s>']
	exp_tree = []
#	textID = []
	#--------- PTP Fix : Add Start/ End and padding token
	##################################
	#parseGT(root, text, ignoreElems)#
	##################################
	exp_tree = parseGT1(root,text, ignoreElems)
#	pdb.set_trace()
	if type(exp_tree[0][0][0]) == type(()):
		exp_tree = exp_tree[0][0]
	else:
		exp_tree = exp_tree[0][0][0]
	text.append('</s>')
	exp_tree.append(('</s>', '</s>'))
	attentionGT = []
	
		
#	print ('gt', len(text))
#	print('exp_tree', len(exp_tree))
	createAttentionGT(attention_list, exp_tree, attentionGT)
	need_to_pad = NetWorkConfig.MAX_TOKEN_LEN - len(text)
	for i in range(need_to_pad):
		attention_map = np.ones((16, 32))
		attentionGT.append(attention_map)
		text.append('$P')
	vector = replaceW2ID(text, word_to_id)
#	print ('gt', len(text))
#	print('exp_tree', len(exp_tree))
#	pdb.set_trace()
#	for i in range(len(attentionGT)-60):
#		resized_attend_map = imresize(attentionGT[i],(256,512))
#		img = im[8].astype(np.float)*resized_attend_map
#		print(img.shape)
#	#ax = plt.subplot(1,2,1)
#	#ax.imshow(attend_map[i])
#		ax = plt.subplot(1,1,1)
#		ax.imshow(img)
#		plt.show()
##	plt.savefig('figures/tmp0_%03d.png' % i)
#		plt.clf()
##	
	
	
#	pdb.set_trace()
	#print (vector)
#	tensor = torch.LongTensor(vector)
#	print('vector',Variable(tensor))
#	return Variable(tensor)
	return im, vector, attentionGT

def getfolSymbol(elem,res):
#	pdb.set_trace()
	if type(elem)!=type([]):
		return elem
	for e in elem:
#		pdb.set_trace()
		if type(e) != type([]) and e[1] not in ['_','^','{','}','[',']']:
			res.append(e)
			return res[0]
		elif type(e) != type([]) and e[1] in ['_','^','{','}','[',']']:
			getfolSymbol(elem[elem.index(e)-1], res)
		else:
			getfolSymbol(e,res)
	
	return res[0]
def getpreSymbol(elem, res):
	if type(elem)!=type([]):
		return elem
	for e in reversed(elem):
#		pdb.set_trace()
		if type(e) != type([]) and e[1] not in ['_','^','{','}','[',']']:
			res.append(e)
			return res[0]
#			break
		elif type(e) != type([]) and e[1] in ['_','^','{','}','[',']']:
			getpreSymbol(elem[elem.index(e)-1], res)
#			break
		else:
			getpreSymbol(e,res)
#			break
	return res[0]

def findEleminExpTree(pattern, exp_tree):
#	expr = ' '.join(str(e) for e in exp_tree if type(e)!=type([]))
#	regex= re.compile(pattern)
#	res = regex.findall(expr)

	res = [e for e in exp_tree if type(e)!=type([]) and pattern in e]
#	pdb.set_trace()
	return res

def normIndexX(index):
	if index < 32:
		return index
	else:
		normIndexX(index-1)
		
def normIndexY(index):
	if index < 16:
		return index
	else:
		normIndexY(index-1)
#print(findEleminExpTree('-_\d', '-_1'))
#		break
#attenion_list = parseOfficialV_4('./../data/TrainINKML/KAIST/KME1G3_0_sub_10.inkml')
#e = getfolSymbol([['-_1', '{', ['\\sqrt_1', '{', 'x_1', '}'], '}', '{', '2_1', '}'], ['d_1', 'x_2']],[])
#e = getfolSymbol(['a_1', ['+_1', ['-_1', '{', '1_1', '}', '{', ['\\sqrt_2', '{', ['a_2', ['+_2', ['-_2', '{', '1_2', '}', '{', ['\\sqrt_3', '{', 'a_3', '}'], '}']]], '}'], '}']]],[])
#print(e)
def createAttentionGT(attention_list, exp_tree, attentionGT):
	for index, elem in enumerate(exp_tree):
#		print('elem', elem)
		if type(elem) != type([]):
			attention_map = np.ones((16, 32))
#			pdb.set_trace()
			if elem[1] in ['_','^','{','}','[',']','</s>']:
				if elem[1] == '_':
					t_pre_symbol = getpreSymbol(exp_tree[index-1],[])
				
					if exp_tree[index+1] == ('{','{'):
						t_fol_symbol = getfolSymbol(exp_tree[index+2],[])
					else:
						t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
					
					pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
					fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
#					pdb.set_trace()
					scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
					scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
				elif elem[1] == '^':
					if ('_','_') in exp_tree:
						t_pre_symbol = getpreSymbol(exp_tree[exp_tree.index(('_','_'))-1],[])
						if exp_tree[index+1] == ('{','{'):
							t_fol_symbol = getfolSymbol(exp_tree[index+2],[])
						else:
							t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
						
						pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
						fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
						scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
						scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
					else:
						t_pre_symbol = getpreSymbol(exp_tree[index-1],[])
						if exp_tree[index+1] == ('{','{'):
							t_fol_symbol = getfolSymbol(exp_tree[index+2],[])
						else:
							t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
						pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
						fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
						scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
						scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
				elif elem[1] == '{':
#					check_sqrt = findEleminExpTree('\\\\sqrt_\d', exp_tree)
#					check_frac = findEleminExpTree('-_\d', exp_tree)
					check_sqrt = findEleminExpTree('\\sqrt', exp_tree)
					check_frac = findEleminExpTree('\\frac', exp_tree)
#					pdb.set_trace()
					if check_sqrt != []:
						t_pre_symbol = check_sqrt[0]
						t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
#						pdb.set_trace()
						pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
						fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
#						pdb.set_trace()
						scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
						scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
					elif check_frac != []:
						t_pre_symbol = check_frac[0]
						t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
						pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
						fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
#						pdb.set_trace()
#						print('pre}', pre_symbol)
						scale_x = pre_symbol[0][2]/16 - int(round((pre_symbol[0][6]-pre_symbol[0][4])/32))
						if index  == exp_tree.index(t_pre_symbol)+1:
							scale_y = pre_symbol[0][3]/16 - int(round((pre_symbol[0][7]-pre_symbol[0][5])/32)) -2
						else:
							scale_y = pre_symbol[0][3]/16 + int(round((pre_symbol[0][7]-pre_symbol[0][5])/32))+2
					else:
#						pdb.set_trace()
						if exp_tree[index-1] == ('_','_'):
							t_pre_symbol = getpreSymbol(exp_tree[index-2],[])
						elif exp_tree[index-1] == ('^','^'):
							if exp_tree[index-2] == ('}','}'):
								t_pre_symbol = getpreSymbol(exp_tree[index-3],[])
							else:
								t_pre_symbol = getpreSymbol(exp_tree[index-2],[])
					
						else:
							t_pre_symbol = getpreSymbol(exp_tree[index-1],[])
						t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
						pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
						fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
#						pdb.set_trace()
						scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
						scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
				elif elem[1] == '[':
					check_sqrt = findEleminExpTree('\\sqrt', exp_tree)
					t_pre_symbol = check_sqrt[0]
					t_fol_symbol = getfolSymbol(exp_tree[index+1],[])
					pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
					fol_symbol = [ele for ele in attention_list if t_fol_symbol[1] in ele]
					scale_x = (pre_symbol[0][2]+fol_symbol[0][2])/32
					scale_y = (pre_symbol[0][3]+fol_symbol[0][3])/32
				else:
					if exp_tree[index-1]==('}','}'):
						t_pre_symbol = getpreSymbol(exp_tree[index-2],[])
					else:
						t_pre_symbol = getpreSymbol(exp_tree[index-1],[])
					pre_symbol = [ele for ele in attention_list if t_pre_symbol[1] in ele]
#					pdb.set_trace()
#					print('pre}', pre_symbol)
#					print('pre}', pre_symbol[0][2])
#					print('pre}', pre_symbol[0][3])
#					if type(fol_symbol == 
					scale_x = (pre_symbol[0][2])/16 +1
					scale_y = (pre_symbol[0][3])/16
					
#				print('x',scale_x)
#				print('y', scale_y)
				scale_x = normIndexX(scale_x)
				scale_y = normIndexY(scale_y)
				attention_map[scale_y, scale_x] = 7
				attention_map = cv2.GaussianBlur(attention_map, (5,5), sigmaX = 2, sigmaY=2)
				attention_map = np.exp(attention_map-np.max(attention_map))/np.sum(np.exp(attention_map-np.max(attention_map)))
			else:
				symbol = [ele for ele in attention_list if elem[1] in ele]
#				pdb.set_trace()
				scale_x = symbol[0][2]/16
				scale_y = symbol[0][3]/16
				dim_1 = int(round((symbol[0][6]-symbol[0][4])/16))
				dim_2 = int(round((symbol[0][7]-symbol[0][5])/16))
				if dim_1%2 == 0:
					dim_1 +=1
				if dim_2%2 == 0:
					dim_2+=1
				attention_map[scale_y, scale_x] = 7
#				print('x',scale_x)
#				print('y', scale_y)
				attention_map = cv2.GaussianBlur(attention_map, (dim_1,dim_2), sigmaX = dim_1/4, sigmaY=dim_2/4)
				attention_map = np.exp(attention_map-np.max(attention_map))/np.sum(np.exp(attention_map-np.max(attention_map)))
			attentionGT.append(attention_map)
#			return attention_map
		else:
			createAttentionGT(attention_list, elem, attentionGT)
#			attentionGT.append(attention_map)
def makeOneshotGT(path_to_ink, path_to_symbol):
	word_to_id, id_to_word = buildVocab(path_to_symbol)
	#print(id_to_word)
	#chuan hoa text de tach ra duoc tung symbol va luu thanh mang trong data
	#TODO
	#data = ['\\forall', 'g', '\\in', 'G'] 
	#print(touchGT(path_to_ink))
	root = getRoot(path_to_ink)
	print('root', root)
	ignoreElems = ['traceFormat','annotation','trace','traceGroup']
	text = ['<s>']
	textid = ['<s>']
	exp_tree= []
	#--------- PTP Fix : Add Start/ End and padding token
	##################################
	#parseGT(root, text, ignoreElems)#
	##################################
#	exp_tree = parseGT1(root, text, textid, ignoreElems)
#	print ('gt', text)
	exp_tree = parseGT1(root, text, ignoreElems)
	print('exp_tree', exp_tree)
	text.append('</s>')
	need_to_pad = NetWorkConfig.MAX_TOKEN_LEN - len(text)
	
	if need_to_pad < 0:
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('WARNING WARNING WARNING WARNING')
		print ('Ground truth size exceed MAX_TOKEN_LEN')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		print ('-------------------------------------------------------------------------------')
		quit()
	
	for i in range(need_to_pad):
		text.append('$P')
	print ('gt', text)
	replaceW2ID(text, word_to_id)
	vector = replaceW2ID(text, word_to_id)
#	print('vector', vector)
	
	#print (vector)
#	tensor = torch.LongTensor(vector)
#	print('vector',Variable(tensor))
#	return Variable(tensor)
	return vector

def getGTfromFolder(input_path, path_to_symbol):
	i=0
	
	for (dirpath, dirnames, filenames) in walk(input_path):
		#pdb.set_trace()
		c = 0
		i = 0
		for file in sorted(filenames):
			print('%d ite of %d' % (i, len(filenames)))
			i += 1
			if not file.endswith('lg'):
				print (input_path + file)
				if checkFile(input_path+file):
					makeGTidVec(input_path + file, path_to_symbol,100,20)
				else:
					c+=1
					#pdb.set_trace()
					print('%d over %d' % (c, len(filenames)))
					continue
				i = i+1
#			if i == 10:
#				break
			
		break
	

def ParseList(toparse_list, padding = 20): #parse by inputed list
	
	ParseResult = []
	GTResult = []
	textGT=[]
	attendGT=numpy.asarray([]).reshape(0, 89, 16, 32)
#	targetRes =[]
#	AttentionGTResult = []

	for filedata in toparse_list:
#		pdb.set_trace()
#		temp_result= parseOfficialV_3(filedata[0], filedata[1][0], padding = padding)

		try:
			#if checkFile(filedata[0]):
			parsed_im,t_textGT, t_attentionGT = makeGTidVec(filedata[0], './parser/mathsymbolclass.txt',filedata[1][0], padding = padding)
			#else:
			#	pass
			#cv2.imshow("big", temp_result[8]); 
			#cv2.waitKey();

		except xml.etree.ElementTree.ParseError as e:
			pdb.set_trace()
		#print('len', len(temp_GT))
		
		#print (filedata[0])
	   
		#print (len(temp_GT))

		if len(parsed_im) == 0:
			print ('unable to parse ' + file)
		else:
			ParseResult.append(parsed_im)
			t_textGT = np.asarray(t_textGT)
			textGT.append(t_textGT)
#			t_attentionGT = np.exp(t_attentionGT-np.max(t_attentionGT))/np.sum(np.exp(t_attentionGT-np.max(t_attentionGT)))
#			pdb.set_trace()
			t_attentionGT = np.expand_dims(t_attentionGT,axis=0)
			#attendGT.append(t_attentionGT)
			try:
				attendGT = numpy.concatenate((attendGT, t_attentionGT), axis=0)
			except ValueError as e:
				pdb.set_trace()
			 
		
	textGT = np.asarray(textGT)
	GTResult.append(textGT)
	GTResult.append(attendGT)
#	pdb.set_trace()
#			targetRes.append(GTResult)
#			AttentionGTResult.append(temp_attentionGT)
	#print (np.asarray(GTResult).shape)
#	pdb.set_trace()
	return np.asarray(ParseResult), GTResult
#a,b = ParseList([('./../data/TrainINKML/KAIST/KME1G3_0_sub_10.inkml', (0.065,0) )])
#getGTfromFolder('./../data/TrainINKML/HAMEX/', './parser/mathsymbolclass.txt')
#makeGTidVec('/Users/ngocbui/ngocbui/Github/LVTN_MER/data/TrainINKML/KAIST/KME1G3_0_sub_10.inkml', './parser/mathsymbolclass.txt',0.065,20)
#makeGTidVec('./../data/TrainINKML/KAIST/KME1G3_1_sub_21.inkml', './parser/mathsymbolclass.txt',0.065,20)
#makeGTidVec('/Users/ngocbui/ngocbui/Github/LVTN_MER/data/TrainINKML/extension/2_em_3.inkml', './parser/mathsymbolclass.txt',1,20)
#makeGTidVec('./../data/miniTrainINKML/MfrDB/MfrDB0001.inkml', './parser/mathsymbolclass.txt',1,20)
#makeGTidVec('./../data/TrainINKML/HAMEX/formulaire001-equation013.inkml', './parser/mathsymbolclass.txt',100,20)
#formulaire001-equation001
#im, attend_map  = parseOfficialV_4('./../data/TrainINKML/KAIST/KME1G3_0_sub_10.inkml',0.065)
