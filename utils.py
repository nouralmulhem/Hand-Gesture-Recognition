# put all imports in this section ######################
import os
import cv2
import numpy as np
# from image_preprocessing import *

from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re

# import pandas as pd
import pickle
from xgboost import XGBClassifier

from Feature_Extraction import *
##########################################################

winSize = (16,16)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9


hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins) # hog of opencv



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
            
            image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
            result=cv2.resize(image, (128, 64)) # multiply by 4
            # pre processing on the image (madbouly)
            # image = Image.open(os.path.join(path, name)).convert('RGB')
            # binary, result = image_pre_processing(image)

            if debug: 
                print("image name = ", name)
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            list_target_names.append(os.path.basename(path))
            list_images.append(result)
    
    return list_target_names,  list_images 

# target_names, images = obtain_images("./data/", True)


def features_extraction(images): 
    list = []
    
    # list = np.array([hog.compute(image)  for image in images])
    all_size=0
    for image in images:   
        # kp, features_list = ORB_features(image)
        shi = shiThomasFeatureExtraction(image, 100, 0.01, 10)
        # hog,_ =hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        
        # print(type(shi))
        # shi=np.asarray(shi)
        # print(type(shi))
        # print (hog)
        # print(shi.shape)
        # print(hog.shape)
        # lbp_feature =lbp(image, radius=3, n_points=8)
        
        list.append(shi)

        # list.append(np.concatenate((hog, shi), axis = None))
        # list.append(hog)
        # features_list, Hog_img = hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # list.append(features_list)
        
    x=min(list,key=len)
    x=len(x)
    features=[]
    for z in list:
        features.append(z[:x])
    features = np.asarray(features)
    return features
    

