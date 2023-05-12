import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from utils import * 

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
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        multichannel=False
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