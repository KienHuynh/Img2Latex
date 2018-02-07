from CROHME_parser import *
from get_gt import *
import glob
import matplotlib.pyplot as plt
from data_augment import *
import numpy as np

import pdb

if __name__ == '__main__':
    np.random.seed(1311)
    data_path = '/home/gvlab/projects/data/CROHME2013/TrainINKML/'
    data_names = ['expressmatch/', 'HAMEX/', 'KAIST/', 'MathBrush/', 'MfrDB/']
    scale_factors = [1.0, 100.0, 0.065, 0.04, 0.8]
    del data_names[0] 
    del scale_factors[0]
    plt.ion()
    plt.figure(figsize=(8,8))
    for s, data_name in enumerate(data_names):
        scale_factor = scale_factors[s]
        sub_data_path = data_path + data_name
        files = glob.glob(sub_data_path + '*.inkml')
        print(data_name)
        for i, f in enumerate(files):
            if (i % 3 != 0):
                continue

            img = inkml2img(files[i], scale_factor)[0]
            plt.subplot(211)
            plt.imshow(img, cmap='gray')

            img = gray2rgb(img)
            #img = random_scale(img, 0.8, 1.2, 10) 
            #img = random_hue(img)
            #img = random_rotate(img, 10)

            img = random_transform(img)
            plt.subplot(212)
            plt.imshow(img)
            plt.pause(0.5)
            
            labels = read_latex_label(files[i], 'mathsymbolclass.txt', 60)
            print('Parsing file %d/%d' % (i, len(files)))
    pdb.set_trace()
