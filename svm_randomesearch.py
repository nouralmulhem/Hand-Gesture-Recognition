from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from utils import *
from FE_Module import *
# Load the hand gesture data and split into training and test sets
print("hello")
load_data(directory='./Dataset_new_filtered/')
X_train, X_test, y_train, y_test, name_train, name_test = read_features()
print("hello2")
# Define the hyperparameter space for SVM
C = [0.1, 1, 10, 100]
kernel = [ 'rbf', 'poly']
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
svm_random = RandomizedSearchCV(estimator = svm, param_distributions = random_grid, n_iter = 150, cv = 5, verbose=2, random_state=0, n_jobs = -1)
svm_random.fit(X_train, y_train)

# Print the best hyperparameters and model accuracy
print(svm_random.best_params_)
print(svm_random.best_score_)
print(svm_random.score(X_test, y_test))
# {'kernel': 'poly', 'gamma': 100.0, 'degree': 4, 'coef0': 0, 'class_weight': 'balanced', 'C': 0.1}
# 'kernel': 'poly', 'gamma': 1000.0, 'degree': 4, 'coef0': 0, 'class_weight': 'balanced', 'C': 0.1