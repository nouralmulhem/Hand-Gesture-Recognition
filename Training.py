from SVM import *
from RandomForest import *
from utils import *
from Performance import *


def failures(ytest, y_pred, name_test, true_class=4, prediction_class=0):
    condition = (ytest != y_pred)

    new_arr = name_test[condition].reshape(-1, 1)
    pred = y_pred[condition].reshape(-1, 1)
    test = ytest[condition].reshape(-1, 1)

    listaia = new_arr[np.logical_and(
        pred == prediction_class, test == true_class)]
    print(listaia.reshape(-1, 1))

    listaia = np.concatenate(
        (np.concatenate((new_arr, pred), axis=1), test), axis=1)

    with open("true.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(listaia)


def classifier(Xtrain, Xtest, ytrain, ytest, name_train, name_test, model_path, show_fails=False):
    model, y_pred = svm_model(Xtrain, Xtest, ytrain)
    calc_accuracy(ytest, y_pred, model, Xtest)
    if show_fails:
        failures(ytest, y_pred, name_test, 5, 0)

    calc_confucion_matrix(ytest, y_pred, no_classes=6)
    pickle.dump(model, open(model_path, 'wb'))

    # model = XGBClassifier(random_state=0)

    # Xtrain = np.array(Xtrain)
    # ytrain = np.array(ytrain)

    # model.fit(Xtrain, ytrain)


Xtrain, Xtest, ytrain, ytest, name_train, name_test = read_features()
classifier(Xtrain, Xtest, ytrain, ytest, name_train,
           name_test, './models/svm2.pkl', show_fails=True)
# classifier(Xtrain, Xtest, ytrain, ytest, './models/random_forest.pkl')
