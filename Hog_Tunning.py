from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

from utils import load_data
# Load the hand gesture data and split into training and validation sets

X_train, X_val, y_train, y_val = load_data(directory='./Dataset/')

# Define the range of parameter values to test
cell_sizes = [(8, 8), (16, 16)]
block_sizes = [(1, 1), (2, 2), (4, 4)]
n_bins = [6, 9, 12]

# Define the SVM classifier

svm = SVC(kernel= 'poly', gamma= 100.0, degree= 4, coef0= 0, class_weight= 'balanced', C= 0.1)

# Define the parameter grid for GridSearchCV
param_grid = {'hog__cell_size': cell_sizes,
              'hog__block_size': block_sizes,
              'hog__orientations': n_bins,
              'svm__C': [0.1, 1, 10, 100],
              'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))}

# Define the pipeline for combining HOG feature extraction and SVM classification
pipeline = Pipeline([('hog', hog()),
                     ('svm', svm)])

# Define the GridSearchCV object and fit to the training data
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and model accuracy on the validation data
print(grid_search.best_params_)
y_pred = grid_search.predict(X_val)
print(accuracy_score(y_val, y_pred))
