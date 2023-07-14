from utils import *
from FE_Techniques import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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


def obtain_images(directory, debug=True):
    list_target_names = []
    list_images = []
    name_files = []
    threads = []
    binariess = []
    index = 0
    print(directory)

    for path, subdirs, files in os.walk(directory):
        if debug:
            print("path = ", path, " number of images = ", len(files))

        for name in files:
            
            pathh = os.path.join(path, name)
            image = Image.open(pathh)
            rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)
            binariess.append(None)
            list_images.append(None)
            binariess.append(None)
            list_images.append(None)
    
            thread = threading.Thread(target=process_image_thread, args=(
                image, index, list_images, binariess))
            thread2 = threading.Thread(target=process_image_thread, args=(
                rotated_img, index+1, list_images, binariess))
            
            threads.append(thread)
            threads.append(thread2)
            
            list_target_names.append(os.path.basename(path))
            name_files.append(name)
            thread.start()
            thread2.start()
            index += 2

            if debug:
                print("image name = ", name)
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
        features_list = hog_features(images[i], orientations=6,pixels_per_cell=(16, 16), cells_per_block=(3, 3))
        # features_list = shiThomasFeatureExtraction(image, 100, 0.01, 10)
        # kp, features_list = SIFT_features(image)

        #  to get fixed number of features using shi and orb and sift
        # features_list = fixed_feature_size(features_list, maxSize)


        # concat the feature list with the true class value and file name just in case of debugging
        list.append(np.concatenate(
            (np.concatenate((features_list, classes[i]), axis=None), files[i]), axis=None))
        

    list = np.asarray(list)
    return list


def load_data(directory = './Dataset_new_filtered/'):
    target_names, images, name_files = obtain_images(directory, True) # load the images and apply image preprocessing
    print("finished obtaining images")
    print(len(images), len(target_names), len(name_files))
    target_names_shuffled, images_shuffled, name_files_shuffled = shuffle(np.array(target_names), np.array(images), np.array(name_files))  # reorder el array only to get different seeds every run
    Xtrain, Xtest, ytrain, ytest, name_train, name_test = train_test_split(
        images_shuffled, target_names_shuffled, name_files_shuffled, random_state=0, test_size=0.2) # split the data set into a testset of 20% and train set of 80%

    train = features_extraction(Xtrain, ytrain, name_train)
    test = features_extraction(Xtest, ytest, name_test)
    print("finished extracting features")

    # write the train feature set and test feature set in csv files
    with open("./features_files/train.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(train)

    with open("./features_files/test.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(test)

# load_data(directory='./Dataset_new_filtered/')
