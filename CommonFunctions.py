# put all imports in this section ######################
import os
import cv2
import numpy as np


from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re

# import pandas as pd
import pickle
from xgboost import XGBClassifier

##########################################################

winSize = (16,16)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9


hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
orb = cv2.ORB_create(nfeatures=10)



def obtain_images(directory, debug = False, prediction_mode = False):
    list_target_names = []
    list_images = []
        
    for path, subdirs, files in os.walk(directory):
        # if(path.startswith(directory + '.')):
        #     continue
        # files = [f for f in files if not f[0] == '.'] # Ignore '.directory' file
        if debug:
            print("path = ",path, " number of images = ", len(files))

        files_list = files        
        
        if len(files) != 0 and prediction_mode:
            files_list = list(sorted( files_list, key=(lambda x:int(x[:-4])) ))

        for name in files_list:                
            image=cv2.imread(os.path.join(path, name))
            
            image=cv2.resize(image, (4*128, 4*64))
            # pre processing on the image (madbouly)

            if debug: 
                print("image name = ", name)
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            list_target_names.append(os.path.basename(path))
            list_images.append(image)
    
    return list_target_names,  list_images 

# target_names, images = obtain_images("./data/", True)


def features_extraction(images): 
    list = []
    
    # list = np.array([hog.compute(image)  for image in images])
    
    for image in images:   
        kp, des = orb.detectAndCompute(image, None)
        list.append(kp)

    list = np.asarray(list)
    return list
    

