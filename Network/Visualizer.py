
import cv2

import sys
sys.path.insert(0, './parser')

use_python_2 = True
if use_python_2:
	import Tkinter
	from Tkinter import *
	import tkFileDialog as filedialog
	
else:
	import tkinter
	from tkinter import *
	from tkinter import filedialog

from PIL import Image, ImageTk
import os



import WAP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import getGT
from torch.autograd import Variable
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import CROHMEParser
import DatasetLoader as DL
import NetWorkConfig

class main_window:
	def __init__(self):

		self.DEBUG = True

		######### CONF #########################

		self.using_cuda = False

		self.word_to_id, self.id_to_word = getGT.buildVocab('./parser/mathsymbolclass.txt')

		######### CONF #########################


		self.model = WAP.WAP()
		self.model.isDebug = True

		self.loader = DL.Loader()

		self.main_handle = Tk()

		self.main_handle.minsize(width=1200, height=600)
		self.main_handle.maxsize(width=1200, height=600)

		
		self.GUIinit()

		self.main_handle.mainloop()
		print ('init')


		
	def GUIinit(self):

		Data_frame = Frame(self.main_handle, width = 20)
		Data_frame.pack()

		Data_Label = Label(Data_frame, text='Model', bd = 5, font=("Consolas", 12))
		Data_Label.pack(side = LEFT)

		self.Model_Entry = Entry(Data_frame, bd =5, width= 80)
		self.Model_Entry.pack(side=LEFT)
		self.Model_Entry.insert(0, '/home/kien/PycharmProjects/LVTN_MER/Network/model/test_model2.mdl')


		btnModelBrowse = Button(Data_frame, text ="Browse", command = self.onBtnDataBrowse)
		btnModelBrowse.pack(side=LEFT)

		Data_Label2 = Label(Data_frame, text='------', bd = 5, font=("Consolas", 12))
		Data_Label2.pack(side = LEFT)

		btnModelLoad = Button(Data_frame, text ="Load", width = 10, command = self.onBtnLoadModel)
		btnModelLoad.pack(side=LEFT)

		########################################################################333

		Label_frame = Frame(self.main_handle, width = 20)
		Label_frame.pack()

		path_Label = Label(Label_frame, text='Path', bd = 5, font=("Consolas", 12))
		path_Label.pack(side = LEFT)

		self.path_Entry = Entry(Label_frame, bd =5, width= 80)
		self.path_Entry.pack(side=LEFT)

		if self.DEBUG:
			self.path_Entry.insert(0, '/home/kien/PycharmProjects/LVTN_MER/Network/model/65_carlos.inkml')

		btnBrowse = Button(Label_frame, text ="Browse", command = self.onBtnBrowse)
		btnBrowse.pack(side=LEFT)

		scale_label = Label(Label_frame, font=("Consolas", 12))
		scale_label.pack(side=LEFT)
		scale_label['text'] = ' || Scale: '
		
		self.scale_Entry = Entry(Label_frame, bd =5, width= 7)
		self.scale_Entry.pack(side=LEFT)
		self.scale_Entry.insert(0, '1.0')
		
		############################################
		
		image_frame = Frame(self.main_handle, height = 200, bd= 10)
		image_frame.pack()
		
		#timg = cv2.imread('myim.png')
		#img = ImageTk.PhotoImage(img)

		#img = ImageTk.PhotoImage(Image.open("myim.png"))

		#self.imgpanel = tkinter.Label(img_frame, image = img)
		#self.imgpanel.pack( fill = "both", expand = "yes")

		#self.img = ImageTk.PhotoImage(Image.open("myim.png"))
		self.panel = Label(image_frame)
		self.panel.pack(side = "bottom", fill = "both", expand = "yes")

		control_frame = Frame(self.main_handle, bd= 10)
		control_frame.pack()
		
		btnPrevious = Button(control_frame, text ="Previous", command = self.onBtnPrev, width = 20)
		btnPrevious.pack(side=LEFT)
		
		btnDebug = Button(control_frame, text ="Debug", command = self.onbtnDebug, width = 20)
		btnDebug.pack(side=LEFT)
		
		btnNext = Button(control_frame, text ="Next", command = self.onBtnNext, width = 20)
		btnNext.pack(side=LEFT)
		
		#########################################################
		
		quickView_frame = Frame(self.main_handle, bd= 10)
		quickView_frame.pack()

		self.predicting_label = Label(quickView_frame, font=("Consolas", 12))
		self.predicting_label.pack(side=LEFT)
		self.predicting_label['text'] = '| Pre/ GT | expect |\n| ?? / ?? | ?????? |'
		
		PreGT_frame = Frame(self.main_handle, bd= 10)
		PreGT_frame.pack()
		
		self.PreGT_label = Label(PreGT_frame, font=("Consolas", 12),justify=LEFT)
		self.PreGT_label.pack(side=TOP)
		self.PreGT_label['text'] = 'GroundTruth: ???????                       \nPredict    : ??????                '
		
	def onBtnLoadModel(self):

		path = self.Model_Entry.get()
		self.model.load_state_dict(torch.load(path))

		print ('load completed')

	def onbtnDebug(self):
		dataset_dat = self.generateDataFromFile(self.path_Entry.get())
		self.train_loader = torch.utils.data.DataLoader(dataset_dat, batch_size=1, shuffle=True)
		self.runAnalyse(0)
		pass
		
	def onBtnBrowse(self):
		file_path = filedialog.askopenfilename()
		self.path_Entry.delete(0, 'end')
		self.path_Entry.insert(0, file_path)

	def onBtnDataBrowse(self):
		file_path = filedialog.askopenfilename()
		self.Model_Entry.delete(0, 'end')
		self.Model_Entry.insert(0, file_path)

		path = self.Model_Entry.get()
		self.model.load_state_dict(torch.load(path))

	def onBtnNext(self):
		self.current_index = self.current_index + 1
		
		if (self.current_index > len(self.Image_list)):
			self.current_index = self.current_index - 1

		self.setIm(self.Image_list[self.current_index])	
		
		print(self.Image_list[self.current_index])

		
	def onBtnPrev(self):
		self.current_index = self.current_index - 1

		if (self.current_index < 0):

			self.current_index = 0

		self.setIm(self.Image_list[self.current_index])	
		
	def setIm(self, im):
		tempImg = ImageTk.PhotoImage(Image.fromarray(im))
		self.panel.configure(image=tempImg)
		self.panel.image = tempImg	

	def generateDataFromFile(self, path):
		self.Image_list = []
		self.current_index = 0
		to_parse = []
		scale = float(self.scale_Entry.get())
		scale_tuple = (scale, 0)
		to_parse.append((path, scale_tuple))

		dataset, target = CROHMEParser.ParseList(to_parse)


		#path2 = 'myim2.png'
		#immm = Image.fromarray(dataset[0,1])
		#img2 = ImageTk.PhotoImage(immm)
		
		self.Image_list.append((dataset[0, 8]))
		self.setIm(self.Image_list[self.current_index])	
		

		dataset_dat = self.loader.generateTensorDatasetFromCROHMENumpy(dataset, target)



		return dataset_dat
		
	def runAnalyse(self, epoch):
		self.model.train()

		for batch_idx, (data, target) in enumerate(self.train_loader):
			

			self.model.setGroundTruth(target.numpy())

			if self.using_cuda:
				data, target = data.cuda(), target.cuda()

			data, target = Variable(data.float()), Variable(target.long())



			output = self.model(data)
			


			
			
			print (type(data))
			target.contiguous()
			output.contiguous()


			target_vec = target.cpu().data.numpy()
			output_vec = numpy.argmax(output.cpu().data.numpy(), axis=2)

			target = target.view(NetWorkConfig.MAX_TOKEN_LEN)
			output = output.view(NetWorkConfig.MAX_TOKEN_LEN, NetWorkConfig.NUM_OF_TOKEN)
			    

			target_vec = target_vec[0][0:30]
			output_vec = output_vec[0][0:30]
			target_str = []
			output_str = []
			for idx in range(0,30):
				target_str.append(self.id_to_word[target_vec[idx]])
				output_str.append(self.id_to_word[output_vec[idx]])
			print('target', ' '.join(target_str))
			print('output', ' '.join(output_str))
				
			self.PreGT_label['text'] = 'GroundTruth: '+ ' '.join(target_str) + '\nPredict    :' + ' '.join(output_str)
			
			#print (self.model.attention_list)
			print (self.model.attention_list[0].shape)
			for i in range(len(self.model.attention_list)):
				temp = self.model.attention_list[i][0]

				
				resized_image = cv2.resize(temp, (512, 256))

				processed_image = resized_image * self.Image_list[0] * 255

				self.Image_list.append(processed_image)

				print ('aaaaaaaaaaaaaaaaaaaa')
				#print (resized_image)


			print(self.model.attention_list)


			#print (len(self.Image_list))

			#print(len(self.Image_list))
			#print(self.Image_list)

a = main_window()
