from utils import *
from FE_Techniques import *
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re
import pickle
import csv
import threading
from PIL import Image
##########################################################

winSize = (16, 16)
blockSize = (8, 8)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9


hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                        cellSize, nbins)  # hog of opencv


def obtain_images(directory, debug=False):
    list_target_names = []
    list_images = []
    name_files = []
    threads = []
    binariess = []
    index = 0

    for path, subdirs, files in os.walk(directory):
        # if(path.startswith(directory + '.')):
        #     continue
        # files = [f for f in files if not f[0] == '.'] # Ignore '.directory' file
        if debug:
            print("path = ", path, " number of images = ", len(files))

        for name in files:
            # image=cv2.imread(os.path.join(path, name))

            # image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
            # result=cv2.resize(image, (128, 64)) # multiply by 4

            # pre processing on the image (madbouly)
            pathh = os.path.join(path, name)
            image = Image.open(pathh)
            thread = threading.Thread(target=process_image_thread, args=(
                image, index, list_images, binariess))
            threads.append(thread)
            binariess.append(None)
            list_images.append(None)
            list_target_names.append(os.path.basename(path))
            name_files.append(name)
            thread.start()
            index += 1
            # image = Image.open(os.path.join(path, name)).convert('RGB')
            # binary, result = image_pre_processing(image)
            # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            if debug:
                print("image name = ", name)
                # cv2.imshow("Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        for thread in threads:
            thread.join()

    return list_target_names, list_images, name_files

# target_names, images = obtain_images("./data/", True)


def fixed_feature_size(list, maxSize):
    feature_vector = np.asarray(list).flatten()
    max_size = min(maxSize, len(feature_vector))
    features = np.zeros((maxSize,))
    features[0:max_size] = feature_vector[0:max_size]
    return features


def features_extraction(images, classes, files):
    list = []
    maxSize = 3000
    # list = np.array([hog.compute(image)  for image in images])
    all_size = 0
    for i in range(len(images)):
        # kp, features_list = orb.detectAndCompute(image, None)
        # features_list =lbp(image, radius=3, n_points=8)
        features = hog_features(images[i], orientations=8,
                                pixels_per_cell=(8, 8), cells_per_block=(3, 3))
        # shi = shiThomasFeatureExtraction(image, 100, 0.01, 10)
        # kp, features_list = SIFT_features(image)

        # feature_vector = features_list.flatten()
        # features = fixed_feature_size(shi, maxSize)
        # features_list = fixed_feature_size(features_list, maxSize)

        # list.append(np.concatenate((features, features_list), axis = None))
        # list.append(shi)
        list.append(np.concatenate(
            (np.concatenate((features, classes[i]), axis=None), files[i]), axis=None))
        # features_list, Hog_img = hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        # list.append(features_list)
    # x = min(list, key=len)
    # x = len(x)
    # features = []
    # for z in list:
    #     features.append(z[:x])
    list = np.asarray(list)
    return list


def load_data(directory):

    target_names, images, name_files = obtain_images(directory)
    target_names_shuffled, images_shuffled, name_files_shuffled = shuffle(np.array(
        target_names), np.array(images), np.array(name_files))  # reorder el array bas
    Xtrain, Xtest, ytrain, ytest, name_train, name_test = train_test_split(
        images_shuffled, target_names_shuffled, name_files_shuffled, random_state=0, test_size=0.2)

    train = features_extraction(Xtrain, ytrain, name_train)
    test = features_extraction(Xtest, ytest, name_test)

    with open("./features_files/train.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(train)

    with open("./features_files/test.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(test)

load_data(directory='./Dataset_new_filtered/')
