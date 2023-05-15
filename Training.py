from SVM import *
from RandomForest import *
from utils import *
from Performance import *


def classifier(Xtrain, Xtest, ytrain, ytest, model_path):
    model, y_pred = svm_model(Xtrain, Xtest, ytrain)
    calc_accuracy(ytest, y_pred, model, Xtest)
    calc_confucion_matrix(ytest, y_pred, no_classes=6)

    # model = XGBClassifier(random_state=0)

    # Xtrain = np.array(Xtrain)
    # ytrain = np.array(ytrain)

    # model.fit(Xtrain, ytrain)

    pickle.dump(model, open(model_path, 'wb'))


# Xtrain, Xtest, ytrain, ytest = load_data(directory='./Dataset/men/0/')
Xtrain, Xtest, ytrain, ytest = load_data(directory='./sora/')
# classifier(Xtrain, Xtest, ytrain, ytest, './models/svm2.pkl')
# classifier(Xtrain, Xtest, ytrain, ytest, './models/random_forest.pkl')
