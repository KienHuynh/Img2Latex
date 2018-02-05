"""
@author: Phuc Phan
Modified by Kien Huynh

Consists functions written to parse the latex XML trees in INKML files into images

These are the approprirate scale factors of equations came from the corresponding sub-datasets (computed using size_statistics):
* expressmatch: 1.0
* HAMEX: 100
* KAIST: 0.065
* MathBrush: 0.04
* MfrDB 0.8
"""

import numpy as np
import cv2
import xml.etree.ElementTree
import os
from os import walk
import get_gt

import pdb


def parse_file(input_path, output_path = 'img.jpg', scale_factor = 1, target_width = 2000, target_height = 1000, vertexlen = 2):
    
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


def size_statistic(input_path, scalefactor = 1):
    """size_statistic
    Compute size statistic of the written equation

    :param input_path: full path to the inkml file
    """

    f = open('statisResult.csv', 'w')
    for (dirpath, dirnames, filenames) in walk(input_path):

        for file in filenames:
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


def inkml2img(input_path, scale_factor = 1, target_width = 512, target_height = 256, padding = 20):
    """inkml2img
    Convert the XML tree in the inkml file into a gray image

    :param input_path: full path to the inkml file
    :param scale_factor: indicating the relative scale of the symbols to the image
    :param target_width: int, width of the image
    :param target_height: int, height of the image
    :param padding: int, number of minimum pixels to be padded (if the equation symbols do not cross the padding margin, no padding will be done)

    :return [output]: list of images
    """
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
            
        for i in range (len(stroke) - 1):
            cv2.line(expr_img, temp_vertex_arr[i], temp_vertex_arr[i + 1], 255, 1)

    #################################
    ##### PADDING ###################
    #################################

    y_offset = int((target_height - expr_img.shape[0]) / 2)
    x_offset = int((target_width - expr_img.shape[1]) / 2)
    
    output[y_offset:y_offset + expr_img.shape[0], x_offset:x_offset + expr_img.shape[1]] = expr_img
 
    return [output]


def prase_list(toparse_list, scale_factors, padding = 20):
    """parse_list
    Parse a list of inkml files into images
    :param toparse_list: list of full paths to inkml files
    :param scale_factors: list of scale factors (float) corresponding to the files in toparse_list
    :param padding: 
    """
    imgs = []
    labels = []
    
    for i, file_path in enumerate(toparse_list):
        temp_img = inkml2img(file_path, scale_factors[i], padding = padding)
        temp_label = get_gt.read_latex_label(file_path, './mathsymbolclass.txt')
       
        if len(temp_img) == 0:
            print ('unable to parse ' + file)
        else:
            imgs.append(temp_img)
            labels.append(temp_label)
    
    return np.asarray(ParseResult), np.asarray(GTResult)
