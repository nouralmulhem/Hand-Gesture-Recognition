from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from skimage.feature import hog

from utils import *

# Load the digits dataset
target,data = tunning_data(directory='./Dataset/')

# Define a pipeline that includes HOG feature extraction, PCA dimensionality reduction,
# and linear SVM classification
pipe = Pipeline([
    ('hog', image.extract_patches_2d),
    ('hog_features', hog),
    ('pca', PCA()),
    ('scaler', StandardScaler()),
    ('svm', svm.LinearSVC())
])

# Define the parameter grid for the HOG parameters and PCA components
param_grid = {
    'hog__patch_size': [(8, 8), (16, 16)],
    'hog__orientations': [8, 9, 10],
    'hog__pixels_per_cell': [(8, 8), (16, 16)],
    'hog__cells_per_block': [(1, 1), (2, 2)],
    'pca__n_components': [16, 32, 64]
}

# Use GridSearchCV to perform a 5-fold cross-validation over the parameter grid
grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the digits dataset
grid_search.fit(data, target)

# Print the best parameters and the corresponding mean cross-validation score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)