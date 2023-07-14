from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn import metrics
from SVM import *

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

    # # Calculate the precision
    # precision = precision_score(y_test, y_pred, pos_label='positive', average='weighted')
    # print('the precision = ', precision*100)

    # # Calculate the recall
    # recall = recall_score(y_test, y_pred, pos_label='positive', average='weighted')
    # print('the recall = ', recall*100)

    # # Calculate the F1 score
    # f1 = f1_score(y_test, y_pred, pos_label='positive', average='weighted')
    # print('the F1 score = ', f1*100)