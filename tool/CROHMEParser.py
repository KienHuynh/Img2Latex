import numpy as np
import cv2
import xml.etree.ElementTree
import os
from os import walk


##########################################################
## parameter:
## input_path: path to inkml file
## output path: path to output
## scale_factor: expression scale (based on inkml file)
## target_width/ height: size of output image
##########################################################
def parseFile(input_path, output_path = 'img.jpg', scale_factor = 100, target_width = 2000, target_height = 1000):
	#################################
	##### GET XML FILE ##############
	#################################

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
			
			for i in range(len(raw_vertex_list) / 2):
				x = float(raw_vertex_list[2 * i])
				y = float(raw_vertex_list[2 * i + 1])

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
	print 'write to ' + output_path


def ParseFolder(input_path, output_path = './', format_ = 'jpg'):

	os.mkdir(output_path + 'ParseResult/')

	for (dirpath, dirnames, filenames) in walk(input_path):
		print dirpath

		for file in filenames:
			temp_filename = file.replace('inkml', format_)
			parseFile(input_path + file, output_path + 'ParseResult/' + temp_filename)

		break


#parseFile('../data/CROHME/test/formulaire007-equation020.inkml', 'img.jpg')
#ParseFolder('../data/CROHME/test/')