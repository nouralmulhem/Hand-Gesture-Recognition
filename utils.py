# put all imports in this section ######################
import os
import cv2
import numpy as np
from image_preprocessing import *

from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re

# import pandas as pd
import pickle
# from xgboost import XGBClassifier

from Feature_Extraction import *

import time
##########################################################

winSize = (16, 16)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9


hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                        cellSize, nbins)  # hog of opencv


def obtain_images(directory, debug=False):
    list_target_names = []
    list_images = []

    for path, subdirs, files in os.walk(directory):
        # if(path.startswith(directory + '.')):
        #     continue
        # files = [f for f in files if not f[0] == '.'] # Ignore '.directory' file
        if debug:
            print("path = ", path, " number of images = ", len(files))

        for name in files:
            # image=cv2.imread(os.path.join(path, name))

            # image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
            # result=cv2.resize(image, (128, 64)) # multiply by 4

            # pre processing on the image (madbouly)
            image = Image.open(os.path.join(path, name)).convert('RGB')
            binary, result = image_pre_processing(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            if debug:
                print("image name = ", name)
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            list_target_names.append(os.path.basename(path))
            list_images.append(result)

    return list_target_names,  list_images

# target_names, images = obtain_images("./data/", True)


def fixed_feature_size(list, maxSize):
    feature_vector = np.asarray(list).flatten()
    max_size = min(maxSize, len(feature_vector))
    features = np.zeros((maxSize,))
    features[0:max_size] = feature_vector[0:max_size]
    return features


def features_extraction(images):
    list = []
    maxSize = 3000
    # list = np.array([hog.compute(image)  for image in images])
    all_size = 0
    for image in images:
        # kp, features_list = orb.detectAndCompute(image, None)
        # features_list =lbp(image, radius=3, n_points=8)
        features = hog_features(image, orientations=9,
                                pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # shi = shiThomasFeatureExtraction(image, 100, 0.01, 10)
        # kp, features_list = SIFT_features(image)

        # feature_vector = features_list.flatten()
        # features = fixed_feature_size(shi, maxSize)
        # features_list = fixed_feature_size(features_list, maxSize)

        # list.append(np.concatenate((features, features_list), axis = None))
        # list.append(shi)
        list.append(features)
        # features_list, Hog_img = hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # list.append(features_list)
    # x = min(list, key=len)
    # x = len(x)
    # features = []
    # for z in list:
    #     features.append(z[:x])
    list = np.asarray(list)
    return list


def load_data(directory):

    target_names, images = obtain_images(directory)
    target_names_shuffled, images_shuffled = shuffle(np.array(target_names), np.array(images))  # reorder el array bas
    Xtrain, Xtest, ytrain, ytest = train_test_split(images_shuffled, target_names_shuffled, random_state=0, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size=0.5, random_state=0)

    Xtrain = features_extraction(Xtrain)
    Xtest = features_extraction(Xtest)

    # n_samples = images_shuffled2.shape[0]

    # print("images_shuffled after : ",len(images_shuffled))
    # print(type(images_shuffled))
    # print(images_shuffled.shape)
    # images_shuffled2 = images_shuffled2.reshape(n_samples, -1)

    return Xtrain, Xtest, ytrain, ytest
    # return Xtrain, Xtest, Xval, ytrain, ytest, yval


def tunning_classifier(directory):

    target_names, images = obtain_images(directory)
    target_names_shuffled, images_shuffled = shuffle(
        np.array(target_names), np.array(images))  # reorder el array bas

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        images_shuffled, target_names_shuffled, random_state=0, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(
        Xtest, ytest, test_size=0.5, random_state=0)

    print(len(X_val))

    X_val = features_extraction(X_val)

    return y_val, X_val


def tunning_feature_extraction(directory):

    target_names, images = obtain_images(directory)
    y, X = shuffle(np.array(target_names), np.array(
        images))  # reorder el array bas

    # x = features_extraction(x)

    return y, X
