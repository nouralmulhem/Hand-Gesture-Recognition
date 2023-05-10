import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from SVM import *

def calc_confucion_matrix(actual, predicted, no_classes):
    classes = list(range(0, no_classes))

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)

    cm_display.plot()
    plt.show()

def calc_accuracy(y_test, y_pred):
    # Evaluate the performance of the SVM model
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy*100)


