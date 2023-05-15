from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from SVM import *
import numpy as np
import csv

def calc_confucion_matrix(actual, predicted, no_classes):
    classes = list(range(0, no_classes))

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)

    cm_display.plot()
    plt.show()

def calc_accuracy(y_test, y_pred, model, Xtest):
    # Evaluate the performance of the SVM model
    accuracy = accuracy_score(y_test, y_pred)
    print('the accuracy = ', accuracy*100, '%')

    # Calculate the precision
    # precision = precision_score(y_test, y_pred)
    # print('the precision = ', accuracy)

    # Calculate the recall
    # recall = recall_score(y_test, y_pred)
    # print('the recall = ', accuracy)

    # # Calculate the F1 score
    # f1 = f1_score(y_test, y_pred)
    # print('the F1 score = ', f1)

    # # Calculate the AUC-ROC score
    # auc_roc = roc_auc_score(y_test, model.predict_proba(Xtest)[:, 1])
    # print('the AUC-ROC score = ', f1)