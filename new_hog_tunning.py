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

# Define the HOG parameters
orientations = [6, 8, 9, 10, 12]  # Different values for orientations
pixels_per_cell = [(8, 8), (16, 16), (32, 32)]  # Different values for pixels per cell
cells_per_block = [(3, 3)]  # Different values for cells per block


list_features = []
list_classes = []
list_names = []
list_acc = []


# Iterate over different parameter combinations
for orientation in orientations:
    for cell_per_block in cells_per_block:
        for pixel_per_cell in pixels_per_cell:
            for i in range(len(list_images)):
                # print(name_files[i])
                # Compute the HOG features
                hog_features = feature.hog(list_images[i], orientations=orientation, pixels_per_cell=pixel_per_cell,
                                            cells_per_block=cell_per_block, visualize=False, transform_sqrt=True,
                                            block_norm='L2-Hys')

                # Perform your desired task with the computed features
                # For example, you can use the features for classification or object detection

                # Print the parameter values and the resulting feature dimensionality
                # print(f"Feature dimensionality: {hog_features.shape[0]}")
                list_features.append(hog_features)
                list_classes.append(list_target_names[i])
                list_names.append(name_files[i])

            list_features = np.asarray(list_features)
            list_classes = np.asarray(list_classes)
            list_names = np.asarray(list_names)
            Xtrain, Xtest, ytrain, ytest, name_train, name_test = train_test_split(list_features, list_classes, list_names, random_state=0, test_size=0.2)

            model, y_pred = svm_model(Xtrain, Xtest, ytrain)
            accuracy = accuracy_score(ytest, y_pred)
            print(f"Orientation: {orientation}, Pixels per cell: {pixel_per_cell}, Cells per block: {cell_per_block}, the accuracy = {accuracy*100}")
            list_acc.append(accuracy)
            
            list_features = []
            list_classes = []
            list_names = []
        
list_acc = np.asarray(list_acc)
print(list_acc.max)
