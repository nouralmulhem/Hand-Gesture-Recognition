import cv2
from skimage import feature
from FE_Module import obtain_images
import numpy as np
from sklearn.model_selection import train_test_split
# from Training import classifier
from SVM import *

print("hello")
list_target_names, list_images, name_files = obtain_images("./Dataset_new_filtered/")
print("hello2")

Xtrain, Xtest, ytrain, ytest, name_train, name_test = train_test_split(list_images, list_target_names, name_files, random_state=0, test_size=0.2)

# Define the HOG parameters
orientations = [6, 8, 9, 10, 12]  # Different values for orientations
pixels_per_cell = [(8, 8), (16, 16)]  # Different values for pixels per cell
cells_per_block = [(1, 1), (2, 2), (3, 3)]  # Different values for cells per block


list_features_train = []
list_classes_train = []
list_names_train = []
list_features_test = []
list_classes_test = []
list_names_test = []
list_acc = []


# Iterate over different parameter combinations
for orientation in orientations:
    for cell_per_block in cells_per_block:
        for pixel_per_cell in pixels_per_cell:
            print(f"to run ==> Orientation: {orientation}, Pixels per cell: {pixel_per_cell}, Cells per block: {cell_per_block}")
            for i in range(len(Xtrain)):
                # print(name_files[i])
                # Compute the HOG features
                hog_features = feature.hog(Xtrain[i], orientations=orientation, pixels_per_cell=pixel_per_cell,
                                            cells_per_block=cell_per_block, visualize=False, transform_sqrt=True,
                                            block_norm='L2-Hys')

                # Perform your desired task with the computed features
                # For example, you can use the features for classification or object detection

                # Print the parameter values and the resulting feature dimensionality
                # print(f"Feature dimensionality: {hog_features.shape[0]}")
                list_features_train.append(hog_features)
                list_classes_train.append(ytrain[i])
                list_names_train.append(name_train[i])
                
            for i in range(len(Xtest)):
                # print(name_files[i])
                # Compute the HOG features
                hog_features = feature.hog(Xtest[i], orientations=orientation, pixels_per_cell=pixel_per_cell,
                                            cells_per_block=cell_per_block, visualize=False, transform_sqrt=True,
                                            block_norm='L2-Hys')

                # Perform your desired task with the computed features
                # For example, you can use the features for classification or object detection

                # Print the parameter values and the resulting feature dimensionality
                # print(f"Feature dimensionality: {hog_features.shape[0]}")
                list_features_test.append(hog_features)
                list_classes_test.append(ytest[i])
                list_names_test.append(name_test[i])

            list_features_train = np.asarray(list_features_train)
            list_classes_train = np.asarray(list_classes_train)
            list_names_train = np.asarray(list_names_train)

            list_features_test = np.asarray(list_features_test)
            list_classes_test = np.asarray(list_classes_test)
            list_names_test = np.asarray(list_names_test)
            # print(list_features, list_classes, list_names)

            model, y_pred = svm_model(list_features_train, list_features_test, list_classes_train)
            accuracy = accuracy_score(list_classes_test, y_pred)
            print(f"the accuracy = {accuracy*100}")
            list_acc.append(accuracy)
            
            list_features_train = []
            list_classes_train = []
            list_names_train = []
            list_features_test = []
            list_classes_test = []
            list_names_test = []
        
list_acc = np.asarray(list_acc)
print(list_acc.max)
