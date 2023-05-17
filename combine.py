from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import *


def combine(Xtrain, Xtest, ytrain, ytest):
    # Train a Random Forest classifier on the training set
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(Xtrain, ytrain)

    # Extract the predicted probabilities of the Random Forest classifier as features
    X_train_rf = rf.predict_proba(Xtrain)
    X_val_rf = rf.predict_proba(Xtest)

    # Concatenate the Random Forest features with the original features
    X_train_combined = np.concatenate([Xtrain, X_train_rf], axis=1)
    X_val_combined = np.concatenate([Xtest, X_val_rf], axis=1)

    # Train an SVM classifier on the combined features
    # svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svm = SVC(kernel='poly', C=0.1, random_state=0, coef0=0,
            degree=4, gamma=100.0, probability=True, class_weight='balanced')
    svm.fit(X_train_combined, ytrain)

    # Evaluate the performance of the combined classifier on the test set
    X_test_rf = rf.predict_proba(Xtest)
    X_test_combined = np.concatenate([Xtest, X_test_rf], axis=1)
    y_test_pred = svm.predict(X_test_combined)
    accuracy = accuracy_score(ytest, y_test_pred)
    print(f"Test accuracy: {accuracy}")

    return svm, rf, y_test_pred