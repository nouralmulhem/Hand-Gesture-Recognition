from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def svm_model(X_train, X_test, y_train):

    # Create an SVM model
    svm = SVC(kernel='poly', C=0.1, random_state=0, coef0=1, degree=4, gamma=10.0,class_weight= None)

    # Train the SVM model on the training set
    svm.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = svm.predict(X_test)

    return svm, y_pred
