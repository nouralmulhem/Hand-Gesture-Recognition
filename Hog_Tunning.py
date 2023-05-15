# import os
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from skimage.feature import hog
# from utils import * 

# # Define the parameters to search over
# param_grid = {
#     'hog__orientations': [9, 10, 11],
#     'hog__pixels_per_cell': [(8, 8), (12, 12)],
#     'hog__cells_per_block': [(2, 2), (3, 3)]
# }

# # Load images from a folder
# X = []
# y = []

# y,X = obtain_images('./Dataset/')
# # for root, dirs, files in os.walk('path/to/folder'):
# #     for file in files:
# #         if file.endswith('.jpg'):
# #             image = cv2.imread(os.path.join(root, file))
# #             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #             resized = cv2.resize(gray, (64, 128))
# #             X.append(resized)
# #             y.append(root.split('/')[-1])  # use the folder name as the class label

# # Convert the images to HOG features
# hog_pipeline = Pipeline([
#     ('hog', hog(
#         orientations=9,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         visualize=False,
#         multichannel=False
#     ))
# ])
# X_hog = hog_pipeline.fit_transform(X)

# # Define the SVM classifier
# svm = SVC(kernel='linear', C=1)

# # Perform GridSearchCV to find the best parameters
# grid_search = GridSearchCV(
#     estimator=svm,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1
# )
# grid_search.fit(X_hog, y)

# # Print the best parameters
# print("Best parameters: ", grid_search.best_params_)



from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils import *



# Define a grid of parameter values to search over
param_grid = {
    'orientations': [6, 9, 12],
    'pixels_per_cell': [(8, 8), (16, 16)],
    'cells_per_block': [(1, 1), (2, 2)]
}

Xtrain, Xtest, ytrain, ytest = load_data(directory='./Dataset/')

# Train an SVM classifier on the HOG features using grid search
svm = SVC(kernel='rbf', C=1, gamma='scale')
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Print the best parameters and validation set performance
print(f"Best parameters: {grid_search.best_params_}")
print(f"Validation accuracy: {grid_search.best_score_}")

# Evaluate the performance of the chosen parameters on the test set
X_test_hog = []
for img in Xtest:
    feature_vector, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    X_test_hog.append(feature_vector)
X_test_hog = np.array(X_test_hog)
y_test_pred = grid_search.predict(X_test_hog)
accuracy = accuracy_score(ytest, y_test_pred)
print(f"Test accuracy: {accuracy}")