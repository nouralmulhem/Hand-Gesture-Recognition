from SVM import *
from RandomForest import *
from utils import *
from Performance import *
from combine import *
from voting import *
import pickle
import csv


def failures(ytest, y_pred, name_test, true_class=4, prediction_class=0, debug = False):
    condition = (ytest != y_pred)

    new_arr = name_test[condition].reshape(-1, 1)
    pred = y_pred[condition].reshape(-1, 1)
    test = ytest[condition].reshape(-1, 1)

    listaia = new_arr[np.logical_and(
        pred == prediction_class, test == true_class)]
    
    if debug:
        print(listaia.reshape(-1, 1))

    listaia = np.concatenate((new_arr, pred), axis=1)

    with open("./results/failures.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(listaia)
        


def classifier_combine(Xtrain, Xtest, ytrain, ytest, name_train, name_test, model_svm_path, model_rf_path, show_fails=False):

    model_svm, model_rf, y_pred = combine(Xtrain, Xtest, ytrain, ytest)
    
    pickle.dump(model_svm, open(model_svm_path, 'wb'))
    pickle.dump(model_rf, open(model_rf_path, 'wb'))

    if show_fails:
        failures(ytest, y_pred, name_test, 5, 0, debug=False)

    calc_accuracy(ytest, y_pred, model_svm, Xtest)
    calc_confucion_matrix(ytest, y_pred, no_classes=6)


def classifier_voting_model(Xtrain, ytrain, name_train,  model_path, show_fails=False):
    model = voting_model(Xtrain,  ytrain)
    pickle.dump(model, open(f'./models/{model_path}', 'wb'))


def classifier(Xtrain, Xtest, ytrain, ytest, name_train, name_test, model_name, show_fails=False):
    model, y_pred = svm_model(Xtrain, Xtest, ytrain)
    pickle.dump(model, open(f'./models/{model_name}', 'wb'))
    
    if show_fails:
        failures(ytest, y_pred, name_test, 5, 0, debug=False)

    calc_accuracy(ytest, y_pred, model, Xtest)
    calc_confucion_matrix(ytest, y_pred, no_classes=6)


# Xtrain, Xtest, ytrain, ytest, name_train, name_test = read_features()
# classifier(Xtrain, Xtest, ytrain, ytest, name_train,
#            name_test, 'svm.pkl', show_fails=True)

# classifier_voting_model(Xtrain,  ytrain, name_train,
#             'voting.pkl', show_fails=True)

# classifier_combine(Xtrain, Xtest, ytrain, ytest, name_train,name_test, './models/svm_combine.pkl', './models/rf_combine.pkl', show_fails=True)
