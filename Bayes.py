#### always keep all your imports in the first cell ####
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
from utils import load_data

def multivariate_normal_gaussian(X, mu, sigma):
    det = np.linalg.det(sigma)
    norm_const = 1 / (((2*np.pi) ** (mu.shape[0]/2)) * (det ** 0.5) )
    x_mu = X - mu
    inv = np.linalg.inv(sigma)
    result = np.exp(-0.5 * (x_mu.T @ inv @ x_mu))
    prob = norm_const * result
    return prob

print("hello bayes before load")
Xtrain, Xtest, ytrain, ytest = load_data("./Dataset/")


numClasses = 6
M = len(Xtrain)
N = len(Xtrain[0]) - 1
K = len(Xtest)

print("hello bayes")

pClasses = [] # A list of size (numClasses, 1) containing the a priori probabilities of each class in the training set.

estimate_means = [] # A numpy array of size (numClasses, N) containing the mean points of each class in the training set. 
                    # HINT: USE NP.MEAN

estimate_covariances = [] # A numpy array of size (numClasses, N, N) containing the covariance matrices of each class in the training set.
                          # HINT: USE NP.COV (Pay attenention for what it takes as an argument)

for classIndex in range(numClasses):
    # TODO [5]: Estimate the parameters of the Gaussian distributions of the given classes.
    # Fill pClasses, estimate_means, and estimate_covariances in this part 
    # Your code should be vectorized WITHOUT USING A SINGLE FOR LOOP.
    pClasses.append(len(ytrain[np.where(ytrain == classIndex+1)])/len(ytrain))
    
    x = np.mean(Xtrain[np.where(ytrain == classIndex+1)] , axis=0)
    estimate_means.append(x)
    
    x = np.cov(Xtrain[np.where(ytrain == classIndex+1)].T)
    estimate_covariances.append(x)
    

estimate_means = np.array(estimate_means)
estimate_covariances = np.array(estimate_covariances)




predicted_classes = np.zeros((K,1)) # predicted_classes: A numpy array of size (K, 1) where K is the number of points in the test set. Every element in this array
                       # contains the predicted class of Bayes classifier for this test point.

for i in range(Xtest.shape[0]):
    print("For test point:", Xtest[i])
    classProbabilities = np.zeros(numClasses)
    # TODO [7.A]: Compute the probability that the test point X_Test[i] belongs to each class in numClasses.
    #  Fill the array classProbabilities accordingly.
    for j in range(3):    
        classProbabilities[j] = ( multivariate_normal_gaussian(Xtest[i], estimate_means[j], estimate_covariances[j]))
        
    # TODO [7.B]: Find the prediction of the test point X_Test[i] and append it to the predicted_classes array.
    predicted_classes[i]=np.argmax(classProbabilities)+1
    


accuracy = len(np.where(ytest == predicted_classes)[0]) / len(predicted_classes)
print('Accuracy = ' + str(round(accuracy,4) * 100) + '%')