from CommonFunctions import *

def predict(Xtest, model_path):
    pickled_model = pickle.load(open(model_path, 'rb'))
    results = pickled_model.predict(Xtest)
    return results

# predict(Xtest, 'model.pkl')


def read_testset(debug = False):
    Xtest = []
    target_names, images = obtain_images("./data/", debug, prediction_mode=True)
    Xtest = features_extraction(images)
    return Xtest

# read_testset(True)

def obtain_results(Xtest, model_path, result_path):
    results = predict(Xtest, model_path)
    with open(result_path, 'w') as f:
        for x in results:
            f.write(str(x))
            f.write('\n')

# obtain_results([1,2,3,4], "data", "results.txt")