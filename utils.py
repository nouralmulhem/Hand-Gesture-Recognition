# put all imports in this section ######################
import os
import cv2
import numpy as np
from image_preprocessing import *

from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import io as ios

# import pandas as pd
import pickle
# from xgboost import XGBClassifier

from Feature_Extraction import *

import time

import csv

# from openpyxl import Workbook
# from openpyxl.drawing.image import Image as Image_EX

from numpy import asarray
from numpy import savetxt

from numpy import loadtxt

import pandas as pd
from PIL import Image
import base64
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
    # target_names_shuffled, images_shuffled = shuffle(np.array(target_names), np.array(images))  # reorder el array bas
    # Xtrain, Xtest, ytrain, ytest = train_test_split(images_shuffled, target_names_shuffled, random_state=0, test_size=0.2)

    # X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size=0.5, random_state=0)

    # wb = Workbook()
    # ws = wb.active

    # for row in Xtrain:
    #     ws.add_image(row)
        
    # wb.save('excel-image.xlsx')
    # with open("new_file.csv","w+") as my_csv:
    #     for data in images:  
    #         # print(data)
    #         # savetxt('data.csv', data, delimiter=',')
    #         csvWriter = csv.writer(my_csv,delimiter=',')
    #         csvWriter.writerows(data)
        
    
        
    # for i in range(172):  
    #     CSVData = open("new_file.csv")
    #     data = np.loadtxt(CSVData, delimiter=",")
    #     # data = loadtxt('data.csv', delimiter=',')
    #     cv2.imshow("data", data)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    # Open CSV file for writing
    # with open('images.csv', 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write header row
    #     writer.writerow(['Image', 'Width', 'Height'])

    #     # Write data rows
    #     for image in images:
    #         width, height = image.shape
    #         print(image.shape)

    #         # Convert image to RGB format and flatten pixel values
    #         # image = image.convert('RGB')
    #         pixels = image.load()
    #         pixel_values = [pixels[x, y] for x in range(width) for y in range(height)]

    #         # Write row to CSV file
    #         writer.writerow(['Image', width, height] + pixel_values)


    # with open('images.csv', 'r', newline='') as csv_file:
    #     reader = csv.reader(csv_file)

    #     # Skip header row
    #     next(reader)

    #     # Loop over data rows
    #     for i, row in enumerate(reader):
    #         # Extract image file path, width, and height
    #         file_path = row[0]

    #         # Use only the images in the list
    #         width = int(row[1])
    #         height = int(row[2])

    #         # Extract flattened pixel values
    #         pixel_values = [int(p) for p in row[3:]]

    #         # Create image object from pixel values
    #         img = Image.new('RGB', (width, height))
    #         pixels = img.load()
    #         for j in range(len(pixel_values)):
    #             x = j % width
    #             y = j // width
    #             pixels[x, y] = tuple(pixel_values[j:j+3])

    #         # Do something with the image
    #         # For example, save it to a file with a different name

    #         cv2.imshow("Image", img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()


    

    
    for image in images:

        # Convert image to base64 encoding
        image_bytes = image.tobytes()
        base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # Create a DataFrame with the image data
        data = pd.DataFrame({'Image': [base64_encoded_image]})

        # Save DataFrame to a CSV file
        csv_file = './file.csv'
        data.to_csv(csv_file, index=False)

        # Load DataFrame from CSV file
        loaded_data = pd.read_csv(csv_file)

        # Retrieve the image from the loaded DataFrame
        base64_encoded_image = loaded_data['Image'][0]
        image_bytes = base64.b64decode(base64_encoded_image.encode('utf-8'))

        # Convert the image bytes to PIL Image object
        loaded_image = Image.open(ios.BytesIO(image_bytes))

        # Display or further process the loaded image
        loaded_image.show()

    Xtrain = features_extraction(Xtrain)
    Xtest = features_extraction(Xtest)



    # with open('file.csv','wb') as out:
    #     csv_out=csv.writer(out)
    #     for row in Xtrain:
    #         csv_out.writerow(row)
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
