from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from utils import load_data

# Load the hand gesture data and split into training and test sets
X_train, X_test, y_train, y_test = load_data(directory='./Dataset/')

# Define the hyperparameter space for SVM
C = [0.1, 1, 10, 100]
kernel = ['linear', 'rbf', 'poly', 'sigmoid']
degree = [2, 3, 4, 5]
gamma = ['scale', 'auto'] + list(np.logspace(-3, 3, 7))
coef0 = [-1, 0, 1]
class_weight = [None, 'balanced']
# Define the SVM classifier
svm = SVC()

# Define the Randomized Search parameters
random_grid = {'C': C,
               'kernel': kernel,
               'degree': degree,
               'gamma': gamma,
               'coef0': coef0,
               'class_weight': class_weight}

# Perform Randomized Search to find the best hyperparameters
svm_random = RandomizedSearchCV(estimator = svm, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
svm_random.fit(X_train, y_train)

# Print the best hyperparameters and model accuracy
print(svm_random.best_params_)
print(svm_random.best_score_)
print(svm_random.score(X_test, y_test))
