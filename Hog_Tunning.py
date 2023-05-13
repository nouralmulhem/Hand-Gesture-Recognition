import os
# import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from utils import * 
from skimage.feature import hog

# Define the parameters to search over
param_grid = {
    'hog__orientations': [9, 10, 11],
    'hog__pixels_per_cell': [(8, 8), (12, 12)],
    'hog__cells_per_block': [(2, 2), (3, 3)]
}

# Load images from a folder
X = []
y = []

y,X = obtain_images('./Dataset/')
# for root, dirs, files in os.walk('path/to/folder'):
#     for file in files:
#         if file.endswith('.jpg'):
#             image = cv2.imread(os.path.join(root, file))
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             resized = cv2.resize(gray, (64, 128))
#             X.append(resized)
#             y.append(root.split('/')[-1])  # use the folder name as the class label

# Convert the images to HOG features
hog_pipeline = Pipeline([
    ('hog', hog(
        np.array(X),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        channel_axis=-1
    ))
])
X_hog = hog_pipeline.fit_transform(X)

# Define the SVM classifier
svm = SVC(kernel='linear', C=1)

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_hog, y)

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)


# from sklearn import datasets, svm
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import image
# from skimage.feature import hog
# from utils import tunning_data

# # Load the digits dataset
# target,data = tunning_data('./Dataset/')

# # Define a pipeline that includes HOG feature extraction, PCA dimensionality reduction,
# # and linear SVM classification
# pipe = Pipeline([
#     ('hog', image.extract_patches_2d),
#     ('hog_features', hog),
#     ('pca', PCA()),
#     ('scaler', StandardScaler()),
#     ('svm', svm.LinearSVC())
# ])

# # Define the parameter grid for the HOG parameters and PCA components
# param_grid = {
#     'hog__patch_size': [(8, 8), (16, 16)],
#     'hog__orientations': [8, 9, 10],
#     'hog__pixels_per_cell': [(8, 8), (16, 16)],
#     'hog__cells_per_block': [(1, 1), (2, 2)],
#     'pca__n_components': [16, 32, 64]
# }

# # Use GridSearchCV to perform a 5-fold cross-validation over the parameter grid
# grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

# # Fit the grid search to the digits dataset
# grid_search.fit(data,target)

# # Print the best parameters and the corresponding mean cross-validation score
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
