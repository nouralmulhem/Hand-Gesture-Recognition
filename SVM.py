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


def svm_model(X_train, X_test, y_train):

    # Create an SVM model
    # svm = SVC(kernel='rbf', C=1, random_state=0)
    svm = SVC(kernel='poly', C=1, random_state=0, coef0=0, degree=4, gamma='scale')

    # Train the SVM model on the training set
    svm.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = svm.predict(X_test)

    return svm, y_pred
