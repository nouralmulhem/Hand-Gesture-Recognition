# put all imports in this section ######################
import os
import cv2
import numpy as np
import pandas as pd

from image_preprocessing import *

import pickle
import time
import threading


def read_features():

    iris_data = pd.read_csv('./features_files/train.csv', sep=',')
    train = np.asarray(iris_data)

    iris_data = pd.read_csv('./features_files/test.csv', sep=',')
    test = np.asarray(iris_data)

    tupple = len(train[0])
    Xtrain = train[:, 0:tupple-2]
    Xtest = test[:, 0:tupple-2]

    ytrain = train[:, tupple-2]
    ytest = test[:, tupple-2]

    name_train = train[:, tupple-1]
    name_test = test[:, tupple-1]

    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain).astype('int')
    ytest = np.asarray(ytest).astype('int')
    name_train = np.asarray(name_train)
    name_test = np.asarray(name_test)

    # for x in ytrain:
    #     print(type(x))

    return Xtrain, Xtest, ytrain, ytest, name_train, name_test


def tunning_classifier(directory):

    Xtrain, Xtest, ytrain, ytest = read_features()

    return ytest, Xtest


def tunning_feature_extraction(directory):

    list_target_names = []
    list_images = []
    threads = []
    binariess = []
    index = 0
    for path, subdirs, files in os.walk(directory):

        for name in files:
            image = Image.open(os.path.join(path, name)).convert('RGB')
            binary, result = image_pre_processing(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            list_target_names.append(os.path.basename(path))
            list_images.append(result)

    return list_target_names, list_images


# read_features()
