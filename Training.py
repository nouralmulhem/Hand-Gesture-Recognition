from SVM import *
from utils import *
from Performance import *


def load_data(directory):

    target_names, images = obtain_images(directory)
    target_names_shuffled, images_shuffled = shuffle(np.array(target_names), np.array(images))  # reorder el array bas
    Xtrain, Xtest, ytrain, ytest = train_test_split(images_shuffled, target_names_shuffled, random_state=0, test_size=0.3)

    Xtrain = features_extraction(Xtrain)
    Xtest = features_extraction(Xtest)

    # n_samples = images_shuffled2.shape[0]

    # print("images_shuffled after : ",len(images_shuffled))
    # print(type(images_shuffled))
    # print(images_shuffled.shape)
    # images_shuffled2 = images_shuffled2.reshape(n_samples, -1)


    return Xtrain, Xtest, ytrain, ytest


# print(Xtrain, ytrain)
# print(len(Xtrain), len(ytrain))
# print(Xtest, ytest)
# print(len(Xtest), len(ytest))


def classifier(Xtrain, Xtest, ytrain, ytest, model_path):
    model, y_pred = svm_model(Xtrain, Xtest, ytrain)
    calc_accuracy(ytest, y_pred)
    calc_confucion_matrix(ytest, y_pred, no_classes=6)

    # model = XGBClassifier(random_state=0)

    # Xtrain = np.array(Xtrain)
    # ytrain = np.array(ytrain)

    # model.fit(Xtrain, ytrain)

    pickle.dump(model, open(model_path, 'wb'))


# Xtrain, Xtest, ytrain, ytest = load_data(directory='./Dataset/')
# classifier(Xtrain, Xtest, ytrain, ytest, './models/svm.pkl')
# classifier(Xtrain, Xtest, ytrain, ytest, './models/random_forest.pkl')




