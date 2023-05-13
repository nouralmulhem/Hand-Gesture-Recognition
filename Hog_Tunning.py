from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from utils import load_data
import numpy as np

class HogTransformer():
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([hog(x.reshape((28, 28)), orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                             cells_per_block=self.cells_per_block) for x in X])
X_train, X_val, y_train, y_val = load_data(directory='./Dataset/')

# Define the parameter grid to search over
param_grid = {
    'orientations': [8, 9, 10],
    'pixels_per_cell': [(8, 8), (10, 10), (12, 12)],
    'cells_per_block': [(2, 2), (3, 3)]
}

# Create a pipeline with the hog feature extractor and the SVM classifier
pipeline = Pipeline([
    ('hog', HogTransformer()),
    ('svm', SVC())
])

# Create a grid search object with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
