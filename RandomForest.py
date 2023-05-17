from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def random_forest(X_train, X_test, y_train):
    # Create a Random Forest model with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    # Train the Random Forest model on the training set
    rf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = rf.predict(X_test)
    
    return rf,y_pred

