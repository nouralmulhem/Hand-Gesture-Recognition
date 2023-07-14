from FE_Module import *
from Training import *
from Prediction import *

import sys

def run(training, path, model_name, debug):
    if(training == 1):
        load_data(path)
        print("finished loading data from images")
        Xtrain, Xtest, ytrain, ytest, name_train, name_test = read_features()
        print("finished reading features")
        if debug == 1:
            classifier(Xtrain, Xtest, ytrain, ytest, name_train, name_test, model_name, show_fails = True)
        else:
            classifier(Xtrain, Xtest, ytrain, ytest, name_train, name_test, model_name, show_fails = False)
        
    elif(training == 2):
        predict(path, model_name, debug = False)
        if debug == 1:
            compare()

run(int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]))