from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def svm_model(X_train, y_train):

    # Create an SVM model
    # svm = SVC(kernel='rbf', C=1, random_state=0)
    # 'kernel': 'poly', 'gamma': 1000.0, 'degree': 4, 'coef0': 0, 'class_weight': 'balanced', 'C': 0.1   84.09
    # {'kernel': 'poly', 'gamma': 10.0, 'degree': 3, 'coef0': 1, 'class_weight': None, 'C': 0.1}
    # {'kernel': 'poly', 'gamma': 0.1, 'degree': 5, 'coef0': 0, 'class_weight': 'balanced', 'C': 1}

    svm = SVC(kernel='poly', C=0.1, random_state=0, coef0=1, degree=4, gamma=10.0,class_weight= None)

    # Train the SVM model on the training set
    svm.fit(X_train, y_train)

    # Make predictions on the testing set
    # y_pred = svm.predict(X_test)

    return svm
