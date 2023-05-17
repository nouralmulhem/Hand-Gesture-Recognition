from utils import *
from FE_Techniques import *
from SVM import *



def predict(debug = False):
    classes_list = []
    time_list = []

    pickled_model = pickle.load(open("./models/svm.pkl", 'rb'))
    # pickled_model_svm = pickle.load(open("./models/svm_combine.pkl", 'rb'))
    # pickled_model_rf = pickle.load(open("./models/rf_combine.pkl", 'rb'))
    
    for path, subdirs, files in os.walk("./test_set_1/"):
        
        test_cases_names = []
        
        if debug:
            print("path = ", path, " number of images = ", len(files))

        files_list = list(sorted(files, key=(lambda x: int(x[:-4]))))

        for name in files_list:
            
            # image=cv2.imread(os.path.join(path, name))

            # image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
            # result=cv2.resize(image, (128, 64)) # multiply by 4

            # pre processing on the image (madbouly)

            test_cases_names.append(os.path.join(path, name))

            # image = Image.open(os.path.join(path, name)).convert('RGB')
            image = Image.open(os.path.join(path, name))
            
            start = time.time()
            
            binary, result = image_pre_processing(image)
            # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            feature_list = hog_features(result, orientations=8,
                                pixels_per_cell=(8, 8), cells_per_block=(3, 3))
            
            # X_test_rf = pickled_model_rf.predict_proba([feature_list])

            # X_train_combined = np.concatenate([[feature_list], X_test_rf], axis=1)
            
            classification = pickled_model.predict([feature_list])
            
            end = time.time()
            difference = round(end - start, 3)
            
            classes_list.append(classification)
            time_list.append(difference)
            

        with open("results.txt", 'w') as f:
            with open("time.txt", 'w') as f2:
                for i in range(len(classes_list)):
                    if debug:
                        f.write(str(test_cases_names[i]))
                        f.write(' ')
                    f.write(str(int(classes_list[i][0])))
                    f2.write(str(time_list[i]))
                    if(i != len(classes_list) - 1):
                        f.write('\n') 
                        f2.write('\n') 

predict(debug = False)


def compare():
    text_file = open("results.txt", "r")
    res_lines = text_file.read().split('\n')
    res_lines = np.asarray(res_lines, dtype=np.int8)
    print(res_lines)
    text_file = open("results_set_1.txt", "r")
    true_lines = text_file.read().split('\n')
    true_lines = np.asarray(true_lines, dtype=np.int8)
    print(true_lines)

    accuracy = accuracy_score(true_lines, res_lines )
    print('the accuracy = ', accuracy*100, '%')

compare()  
