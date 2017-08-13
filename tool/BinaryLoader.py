import cv2
import numpy as np
import matplotlib.pyplot as plt
#
# Use python 3.x
#

def readBinaryFile(path):
	loaded_data = np.load(path)
	print (loaded_data.shape)
	print (loaded_data.dtype)
	plt.imshow(loaded_data[5][7])
	plt.show()

	#cv2.imshow('im', loaded_data[5][8])
	cv2.waitKey(0)

	

	pass

readBinaryFile('./../data/CROHME/Binary/CROHMEBLOCK_Data.npy')
