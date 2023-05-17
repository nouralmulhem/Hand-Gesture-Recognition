from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from utils import read_features
# from FE_Module import load_data

# Load the iris dataset and scale the features
# print("hello")
# load_data(directory='./Dataset_new_filtered/')
# Xtrain, Xtest, ytrain, ytest, name_train, name_test = read_features()
# print("hello2")

def voting_model(Xtrain,  ytrain):
    # Initialize the individual classifiers
    knn = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(kernel='poly', C=0.1, random_state=0, coef0=1, degree=4, gamma=10.0,class_weight= None, probability=True)
    rf = RandomForestClassifier( max_depth= None, min_samples_leaf= 1, min_samples_split= 5,n_estimators=100)
    # Create the voting classifier
    voting_clf = VotingClassifier(
        estimators=[('knn', knn), ('svm', svm), ('rf', rf)],
        voting='soft' 
        ,weights=[1.4,1.5,1.3]
    )

    # Train the voting classifier
    voting_clf.fit(Xtrain, ytrain)

    # Make predictions
    # y_pred = voting_clf.predict(Xtest)
    return voting_clf 
# Calculate accuracy
# accuracy = accuracy_score(ytest, y_pred)
# print("Voting Classifier Accuracy:", accuracy)
