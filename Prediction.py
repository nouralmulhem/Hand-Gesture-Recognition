from utils import *
from FE_Techniques import *
from SVM import *
import time
import pickle

def predict(path = './test_true/', model_name = 'svm.pkl', debug = False):
    classes_list = []
    time_list = []

    pickled_model = pickle.load(open(f'./models/{model_name}', 'rb'))

    for path, subdirs, files in os.walk(path):
        
        test_cases_names = []
        
        if debug:
            print("path = ", path, " number of images = ", len(files))

        files_list = list(sorted(files, key=(lambda x: int(x[:-4]))))

        for name in files_list:

            test_cases_names.append(os.path.join(path, name))

            image = Image.open(os.path.join(path, name))
            
            start = time.time()
            
            binary, result = image_pre_processing(image)
            
            feature_list = hog_features(result, orientations=6,
                                pixels_per_cell=(16, 16), cells_per_block=(3, 3))
            
            classification = pickled_model.predict([feature_list])
            
            end = time.time()
            difference = round(end - start, 3)
            
            classes_list.append(classification)
            time_list.append(difference)
            

        with open("./results/results.txt", 'w') as f:
            with open("./results/time.txt", 'w') as f2:
                for i in range(len(classes_list)):
                    if debug:
                        f.write(str(test_cases_names[i]))
                        f.write(' ')
                    f.write(str(int(classes_list[i][0])))
                    f2.write(str(time_list[i]))
                    if(i != len(classes_list) - 1):
                        f.write('\n') 
                        f2.write('\n') 



def compare(compare_with = './results/test_true.txt'):
    text_file = open("./results/results.txt", "r")
    res_lines = text_file.read().split('\n')
    res_lines = np.asarray(res_lines, dtype=np.int8)

    text_file = open(compare_with, "r")
    true_lines = text_file.read().split('\n')
    true_lines = np.asarray(true_lines, dtype=np.int8)

    accuracy = accuracy_score(true_lines, res_lines )
    print('the accuracy = ', accuracy*100, '%')

predict(path = './test_true/', model_name = 'voting.pkl', debug = False)
compare()
