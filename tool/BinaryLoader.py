import cv2
import numpy as np

#
# Use python 3.x
#

def readBinaryFile(path):
	loaded_data = np.load(path)
	print (loaded_data.shape)
	cv2.imshow('im', loaded_data[5][0])
	cv2.waitKey(0)

	

	pass

readBinaryFile('./../data/CROHME/Binary/CROHMEBLOCK_Data.npy')
