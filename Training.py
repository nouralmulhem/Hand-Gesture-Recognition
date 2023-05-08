from CommonFunctions import *


def load_data(directory):

    target_names, images = obtain_images(directory)
    
    target_names_shuffled, images_shuffled = shuffle(np.array(target_names), np.array(images)) # reorder el array bas

    n_samples= images_shuffled.shape[0]
    
    images_shuffled2 = features_extraction(images_shuffled)
    images_shuffled2 = images_shuffled2.reshape(n_samples,-1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(images_shuffled2, target_names_shuffled, random_state=0, test_size=0.2)

    return Xtrain, Xtest, ytrain, ytest 
    

Xtrain, Xtest, ytrain, ytest =load_data(directory= './Dataset/')    
print(Xtrain, ytrain)
print(len(Xtrain), len(ytrain))
print(Xtest, ytest)
print(len(Xtest), len(ytest))



def classifier(Xtrain, Xtest, ytrain, ytest, model_path):
    model = None # some model
    
    # model = XGBClassifier(random_state=0)
    
    # Xtrain = np.array(Xtrain)
    # ytrain = np.array(ytrain)
    
    # model.fit(Xtrain, ytrain)
    
    pickle.dump(model, open(model_path, 'wb'))

classifier(Xtrain, Xtest, ytrain, ytest, 'model.pkl')
