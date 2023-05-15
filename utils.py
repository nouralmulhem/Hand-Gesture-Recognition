# put all imports in this section ######################
import os
import cv2
import numpy as np

from image_preprocessing import *


def read_features():
    CSVData = open("./features_files/Xtrain.csv")
    Xtrain = np.loadtxt(CSVData, delimiter=",")
    
    CSVData = open("./features_files/Xtest.csv")
    Xtest = np.loadtxt(CSVData, delimiter=",")
    
    CSVData = open("./features_files/ytrain.csv")
    ytrain = np.loadtxt(CSVData, delimiter=",")
    
    CSVData = open("./features_files/ytest.csv")
    ytest = np.loadtxt(CSVData, delimiter=",")

    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain)
    ytest = np.asarray(ytest)

    return Xtrain, Xtest, ytrain, ytest


def tunning_classifier(directory):
    
    Xtrain, Xtest, ytrain, ytest = read_features()

    return ytest, Xtest



def tunning_feature_extraction(directory):

    list_target_names = []
    list_images = []

    for path, subdirs, files in os.walk(directory):

        for name in files:
            image = Image.open(os.path.join(path, name)).convert('RGB')
            binary, result = image_pre_processing(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            list_target_names.append(os.path.basename(path))
            list_images.append(result)

    return list_target_names, list_images


