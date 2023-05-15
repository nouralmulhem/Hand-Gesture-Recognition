from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from utils import load_data

# Load the hand gesture data and split into training and test sets
X_train, X_test, y_train, y_test = load_data(directory='./Dataset/')

# Define the hyperparameter space for Random Forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
max_features = ['sqrt', 'log2', None]
max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Define the Random Forest classifier
rf = RandomForestClassifier()

# Define the Randomized Search parameters
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Perform Randomized Search to find the best hyperparameters
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 400, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

# Print the best hyperparameters and model accuracy
print(rf_random.best_params_)
print(rf_random.best_score_)
print(rf_random.score(X_test, y_test))
