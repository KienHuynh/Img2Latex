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
import pdb

##########################################################
## parameter:
## input_path: path to inkml file
## output path: path to output
## scale_factor: expression scale (based on inkml file)
## target_width/ height: size of output image
##########################################################

#
#
#
# HAMEX: 100
# KAIST: 0.065
# MathBrush: 0.04
# MfrDB 0.8

def parseFile(input_path, output_path = 'img.jpg', scale_factor = 1, target_width = 2000, target_height = 1000, vertexlen = 2):
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path

	root = xml.etree.ElementTree.parse(input_path).getroot()
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
			processing_text = processing_text.replace(',', '')
			processing_text = processing_text.replace('\n', '')
			raw_vertex_list = processing_text.split(' ')
			
			for i in range(len(raw_vertex_list) / vertexlen):
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

	expr_img = np.zeros((int(heigh * scale_factor) + 1 , int(width * scale_factor) + 1 ))

	for stroke in vertex_arr:

		temp_vertex_arr = []

		for vertex in stroke:
			temp_vertex_arr.append((int((vertex[0] - min_x) * scale_factor ), int((vertex[1] - min_y) * scale_factor)))
			
		for i in range (len(stroke) - 1):
			cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)

	#################################
	##### PADDING ###################
	#################################
		
	y_offset = (target_height - expr_img.shape[0]) / 2  
	x_offset = (target_width - expr_img.shape[1]) / 2  
	output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img

	#cv2.imshow("big", output); Image Window will be clipped if its size is bigger than screen's size!
	#cv2.waitKey();
	cv2.imwrite(output_path, output)
	print ('write to ' + output_path)

def sizeStatistic(input_path, scalefactor = 1):

	f = open('statisResult.csv', 'w')
	for (dirpath, dirnames, filenames) in walk(input_path):

		for file in filenames:
			#parseFile(input_path + file, output_path + 'ParseResult/' + temp_filename)
			print (file)

			try:
				root = xml.etree.ElementTree.parse(input_path + file).getroot()
			except:
				print ('error parsing')
				continue

			tag_header_len = len(root.tag) - 3

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
							
					for i in range(len(raw_vertex_list) / vertexlen):
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

			
			min_x = min_x * scalefactor
			max_x = max_x * scalefactor
			min_y = min_y * scalefactor
			max_y = max_y * scalefactor
			print (max_y)
			print (min_y)	
			print (min_x)	
			print (max_x)
			f.write(str(min_x) + ',' + str(max_x) + ',' + str(min_y) + ',' + str(max_y) + '\n')


	f.close()		

# Unknow vertex size
def parseFileSpecial(input_path, output_path = 'img.jpg', scale_factor = 1, target_width = 2000, target_height = 1000, padding = 0):
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path

	try:
		root = xml.etree.ElementTree.parse(input_path).getroot()
	except:
		print ('error parsing')
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
			
			
			for i in range(len(raw_vertex_list) / vertexlen):
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

	expr_img = np.zeros((int(heigh * scale_factor) + 1 , int(width * scale_factor) + 1 ))

	for stroke in vertex_arr:

		temp_vertex_arr = []

		for vertex in stroke:
			temp_vertex_arr.append((int((vertex[0] - min_x) * scale_factor ), int((vertex[1] - min_y) * scale_factor)))
			
		for i in range (len(stroke) - 1):
			cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)

	#################################
	##### PADDING ###################
	#################################
		
	y_offset = (target_height - expr_img.shape[0]) / 2  
	x_offset = (target_width - expr_img.shape[1]) / 2  
	output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img

	#cv2.imshow("big", output); Image Window will be clipped if its size is bigger than screen's size!
	#cv2.waitKey();
	cv2.imwrite(output_path, output)
	print ('write to ' + output_path)


######################################## parse scale size
def parseOfficialV_1(input_path, output_path = 'img.jpg', scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path

	try:

		try:
			root = xml.etree.ElementTree.parse(input_path).getroot()
		except:
			print ('error parsing')
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
				
				
				for i in range(len(raw_vertex_list) / vertexlen):
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

			temp_vertex_arr = []

			for vertex in stroke:
				temp_vertex_arr.append((int((vertex[0] - min_x) * scale_factor ), int((vertex[1] - min_y) * scale_factor)))
				
			for i in range (len(stroke) - 1):
				cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)

		#################################
		##### PADDING ###################
		#################################
			
		y_offset = (target_height - expr_img.shape[0]) / 2  
		x_offset = (target_width - expr_img.shape[1]) / 2  
		

		output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img

		#cv2.imshow("big", output); Image Window will be clipped if its size is bigger than screen's size!
		#cv2.waitKey();
		cv2.imwrite(output_path, output)
		print ('write to ' + output_path)
	except Exception as e:
		print (e)
		print ('error parse ' + input_path)
		exit()

######################################## to bin
def parseOfficialV_2(input_path, scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path

	try:

		try:
			root = xml.etree.ElementTree.parse(input_path).getroot()
		except:
			print ('error parsing')
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
				
				
				for i in range(len(raw_vertex_list) / vertexlen):
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

			temp_vertex_arr = []

			for vertex in stroke:
				temp_vertex_arr.append((int((vertex[0] - min_x) * scale_factor ), int((vertex[1] - min_y) * scale_factor)))
				
			for i in range (len(stroke) - 1):
				cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)

		#################################
		##### PADDING ###################
		#################################
			
		y_offset = (target_height - expr_img.shape[0]) / 2  
		x_offset = (target_width - expr_img.shape[1]) / 2  
		

		output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img

		#cv2.imshow("big", output); Image Window will be clipped if its size is bigger than screen's size!
		#cv2.waitKey();
		#cv2.imwrite(output_path, output)
		
		return [output]
	except Exception as e:
		print (e)
		print ('error parse ' + input_path)
		return np.array([])

######################################## to python 3.6
def parseOfficialV_3(input_path, scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
	
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path
	try:
		try:
			root = xml.etree.ElementTree.parse(input_path).getroot()
		except:
			print ('error parsing')
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
		
#		cv2.imshow("big3", output); #Image Window will be clipped if its size is bigger than screen's size!
#		cv2.waitKey();
		#cv2.imwrite(output_path, output)
		#print('dacoanh')
		eight_imgs = []
		for theta in np.arange(0,2*np.pi, np.pi / 4):
			
			kernel = cv2.getGaborKernel((30,30), 3.0, theta, 5.0, 0.5, 0, cv2.CV_32F)
#			print('kernel', kernel)
			filtered_img = cv2.filter2D(output, cv2.CV_8UC3, kernel)
#			cv2.imshow('filtered image', filtered_img)
#			cv2.waitKey();
#			break 
#			print('fitered_img', filtered_img)
			eight_imgs.append(filtered_img)
		eight_imgs.append(output) 
		
#		print('type', eight_imgs)
#			cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
		return eight_imgs
	except NotImplementedError as e:
		print (e)
		print ('error parse ' + input_path)
		return np.array([])
	
def parseOfficialV_4(input_path, groundtruth, scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
	
	#################################
	##### GET XML FILE ##############
	#################################
	#print 'processing ' + input_path
	try:
		try:
			root = xml.etree.ElementTree.parse(input_path).getroot()
		except:
			print ('error parsing')
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


		word_to_id, id_to_word = getGT.buildVocab('./mathsymbolclass.txt')

		for child in root:
			tag = child.tag[tag_header_len:]
			if tag == 'traceGroup':
				traceGroupRoot = child
				for TGchild in traceGroupRoot:

					print('=================')

					trace_list = []

					sum_of_pts_x = 0
					sum_of_pts_y = 0
					pts_count = 0

					for TGGrandChild in TGchild:

						childtag = TGGrandChild.tag[tag_header_len:]
						#print(childtag)

						if childtag == 'annotationXML':
							href = TGGrandChild.get('href')

						if childtag == 'annotation':
							anno = TGGrandChild.text

						if childtag == 'traceView':
							trace_id = int(TGGrandChild.get('traceDataRef'))

							for i in range(len(vertex_arr[trace_id])):

								#print ((vertex_arr[trace_id][i][0] - min_x)*scale_factor)

								sum_of_pts_x += vertex_arr[trace_id][i][0] - min_x
								sum_of_pts_y += vertex_arr[trace_id][i][1] - min_y
				
							pts_count += len(vertex_arr[trace_id])

							
					if pts_count > 0:
						av_x = sum_of_pts_x * scale_factor / pts_count + x_offset
						av_y = sum_of_pts_y * scale_factor / pts_count + y_offset
						retdata = (anno, href, int(av_x), int(av_y))
						
						attention_list.append(retdata)
		print(attention_list)

		attentionGT = createAttentionGT(attention_list, input_path, './mathsymbolclass.txt')


		
		return eight_imgs, attentionGT
	except NotImplementedError as e:
		print (e)
		print ('error parse ' + input_path)
		return np.array([])


def createAttentionGT(centerofSymbol_list, path_to_ink, path_to_label):
	gtID = getGT.makeGTidVec(path_to_ink, path_to_label)
	invisible_symbols = ['_','^','{','}','[',']','</s>']
	attention_block =[]
	
	temp_x = 0
	temp_y = 0
	previous_symbol = '<s>'
	is_bracket_first = 1
	for elem in gtID:
		attention_map = np.ones((16, 32))
		is_exist = 0 
		for item in centerofSymbol_list:
			if item[1] == elem:
				scale_x = item[2]/16
				scale_y = item[3]/16
				attention_map[scale_y, scale_x] = 7
				temp_x = scale_x
				temp_y = scale_y
				previous_symbol = item[0]
				is_exist = 1
				break 
				
		if is_exist == 0:
			if elem in invisible_symbols:
				if elem == '_':
					scale_x = temp_x + 1
					scale_y = temp_y + 1

				elif elem == '^':
					scale_x = temp_x + 1
					scale_y = temp_y - 1

				elif elem == '{':
					if previous_symbol == '_':
						scale_x = temp_x + 1
						scale_y = temp_y - 1
						
					elif previous_symbol == '^':
						scale_x = temp_x + 1
						scale_y = temp_y + 1
					elif previous_symbol == 'frac' and is_bracket_first:
						scale_x = temp_x - 1 #len(frac)/2
						scale_y = temp_y - 1
						is_bracket_first = 0
					elif previous_symbol == 'frac' and not is_bracket_first:
						scale_x = temp_x - 1 #len(frac)/2
						scale_y = temp_y + 1
						is_bracket_first = 1
					else:
						scale_x = temp_x +1
						scale_y = temp_y
				elif elem == '[':
					scale_x = temp_x - 1
					scale_y = temp_y - 1 
				else:
					scale_x = temp_x +1
					scale_y = temp_y
					
				previous_symbol = elem
				attention_map[scale_y, scale_x] = 7
				temp_x = scale_x
				temp_y = scale_y
			elif elem == '$P':
				pass 
			else:
				print('missing invisible elem', elem)
				
		attention_block.append(attention_map)
	return attention_block
	
def ParseFolder(input_path, scale_factor = 1, output_path = './', format_ = 'jpg', verlen = 2, padding = 20):
	try:
		os.mkdir(output_path + 'ParseResult/')
	except:
		pass

	count = 0

	for (dirpath, dirnames, filenames) in walk(input_path):
		print (dirpath)

		for file in filenames:
			temp_filename = file.replace('inkml', format_)
			#parseFile(input_path + file, output_path + 'ParseResult/' + temp_filename, scale_factor, vertexlen = verlen)
			parseOfficialV_1(input_path + file, output_path + 'ParseResult/' + temp_filename, scale_factor, padding = padding)

			count = count + 1
			if count == 100:
				break

		break

def ParseFolderToBinary(input_path, scale_factor = 1, output_path = './', verlen = 2, padding = 20):
	try:
		os.mkdir(output_path + 'ParseResult/')
	except:
		pass

	count = 0

	ParseResult = []
	GTResult = []
	real_output_path_Data = output_path + 'ParseResult/' + 'CROHMEBLOCK_Data'
	real_output_path_Target = output_path + 'ParseResult/' + 'CROHMEBLOCK_Target'

	for (dirpath, dirnames, filenames) in walk(input_path):
		for file in filenames:
			temp_result = parseOfficialV_3(input_path + file, scale_factor, padding = padding)
			temp_GT = getGT.prepareTarget(getGT.makeOneshotGT(input_path + file, './mathsymbolclass.txt'))

			print (file)
			#print (len(temp_GT))
			
			if len(temp_result) == 0:
				print ('unable to parse ' + file)
			else:
				ParseResult.append(temp_result)
				GTResult.append(temp_GT)

			count = count + 1
			if count == 100:
				break

		break

	np.save(real_output_path_Target, GTResult)
	np.save(real_output_path_Data, ParseResult)
	
	#tempppp1 = np.load(real_output_path_Data + '.npy')
	#tempppp2 = np.load(real_output_path_Target + '.npy')
	#
	#print (tempppp1.shape)
	#print (tempppp2)
	#print (numpy.asarray(tempppp2).shape)
	#Tensor_train = torch.utils.data.TensorDataset(torch.from_numpy(tempppp1), torch.from_numpy(tempppp2.astype(numpy.long)))

def ParseFolderToBinary2(input_path, scale_factor = 1, output_path = './', padding = 20):
	try:
		os.mkdir(output_path + 'ParseResult/')
	except:
		pass
	
	count = 0
	
	ParseResult = []
	GTResult = []
	real_output_path_Data = output_path + 'ParseResult/' + 'CROHMEBLOCK_Data'
	real_output_path_Target = output_path + 'ParseResult/' + 'CROHMEBLOCK_Target'
	
	for (dirpath, dirnames, filenames) in walk(input_path):
		for file in filenames:
			temp_result = parseOfficialV_3(input_path + file, scale_factor, padding = padding)
			temp_GT = getGT.makeOneshotGT(input_path + file, './mathsymbolclass.txt')
			print (file)
		   
			if len(temp_result) == 0:
				print ('unable to parse ' + file)
			else:
				ParseResult.append(temp_result)
				GTResult.append(temp_GT)
				
			count = count + 1
			if count == 2:
				break
			
		break
	
	#print(np.asarray(ParseResult).shape)

	np.save(real_output_path_Target, GTResult)
	np.save(real_output_path_Data, ParseResult)
	
	#tempppp1 = np.load(real_output_path_Data + '.npy')
	#tempppp2 = np.load(real_output_path_Target + '.npy')
	#
	#print (tempppp1.shape)
	#print (tempppp2)
	#print (numpy.asarray(tempppp2).shape)
	#Tensor_train = torch.utils.data.TensorDataset(torch.from_numpy(tempppp1), torch.from_numpy(tempppp2.astype(numpy.long)))


def ParseList(toparse_list, padding = 20): #parse by inputed list
	
	ParseResult = []
	GTResult = []
	

	for filedata in toparse_list:
		temp_result = parseOfficialV_3(filedata[0], filedata[1][0], padding = padding)
		try:
			temp_GT = getGT.makeOneshotGT(filedata[0], './parser/mathsymbolclass.txt')

			#cv2.imshow("big", temp_result[8]); 
			#cv2.waitKey();

		except xml.etree.ElementTree.ParseError as e:
			pdb.set_trace()
		#print('len', len(temp_GT))
		
		#print (filedata[0])
	   
		#print (len(temp_GT))

		if len(temp_result) == 0:
			print ('unable to parse ' + file)
		else:
			ParseResult.append(temp_result)
			GTResult.append(temp_GT)
	
	#print (np.asarray(GTResult).shape)

	return np.asarray(ParseResult), np.asarray(GTResult)

	#np.save(real_output_path_Target, GTResult)
	#np.save(real_output_path_Data, ParseResult)
#im, attend_map  = parseOfficialV_4('./../../data/miniTrainINKML/expressmatch/106_Fabricio.inkml', getGT.makeGTidVec('./../../data/miniTrainINKML/expressmatch/106_Fabricio.inkml', './mathsymbolclass.txt'))
#print(attend_map[0])
#cv2.imshow('a', im[8])
#cv2.waitKey()
#parseOfficialV_3('./../data/Tra inINKML/expressmatch/101_fujita.inkml')
#parseFileSpecial('./TrainINKML/TrainINKML/MfrDB/MfrDB0104.inkml', 'img.jpg')
#ParseFolder('./../data/TrainINKML/expressmatch/', 1, verlen = 2, output_path = 'expressResult/', padding = 50)
#ParseFolder('./TrainINKML/TrainINKML/KAIST/', 0.065, verlen = 2, output_path = 'expressResult/', padding = 50)
#ParseFolder('./TrainINKML/TrainINKML/MfrDB/', 0.8, verlen = 2, output_path = 'expressResult/', padding = 50)
#ParseFolderToBinary2('./testFolder/', 1, output_path = './', padding = 50)
#ParseList([('./../../data/TrainINKML/MfrDB/MfrDB0104.inkml', (1,0) )])
#ParseFolder('./TrainINKML/TrainINKML/HAMEX/', 100, verlen = 2, output_path = 'expressResult/', padding = 50)
#ParseFolder('./TrainINKML/TrainINKML/expressmatch/', 1, verlen = 2)

#sizeStatistic('./TrainINKML/TrainINKML/MfrDB/', 0.8)

# expressmatch
# HAMEX: 100
# KAIST: 0.065
# MathBrush: 0.04
# MfrDB 0.8
