from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create a Random Forest model with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the Random Forest model on the training set
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the performance of the Random Forest model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
