from utils import *
from FE_Techniques import *

import pickle
import time

def predict(debug = False):
    classes_list = []
    time_list = []

    pickled_model = pickle.load(open("./models/svm.pkl", 'rb'))
    
    for path, subdirs, files in os.walk("./data/"):
        
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

            image = Image.open(os.path.join(path, name)).convert('RGB')
            
            start = time.time()
            
            binary, result = image_pre_processing(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            
            feature_list = hog_features(result, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            
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
                    f.write('\n') 
                    f2.write(str(time_list[i]))
                    f2.write('\n') 

predict(debug = True)