#### always keep all your imports in the first cell ####
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
from Training import load_data

def multivariate_normal_gaussian(X, mu, sigma):
    z,_=sigma.shape
    prob = (1/(((2*np.pi)**(z/2))*(np.linalg.det(sigma)**0.5)))* np.exp(-0.5*(np.dot(np.dot((X-mu).T,np.linalg.inv(sigma)),(X-mu))))
    return prob

print("hello bayes before load")
Xtrain, Xtest, ytrain, ytest = load_data("./Dataset/")


numClasses = 6
M = len(Xtrain)
N = len(Xtrain[0]) - 1
K = len(Xtest)
ytrain=np.asarray(ytrain)
print("hello bayes")

pClasses =np.zeros((numClasses, 1)) # A list of size (numClasses, 1) containing the a priori probabilities of each class in the training set.

estimate_means = np.zeros((numClasses,N)) # A numpy array of size (numClasses, N) containing the mean points of each class in the training set. 
                    # HINT: USE NP.MEAN

estimate_covariances = np.zeros((numClasses,N,N)) # A numpy array of size (numClasses, N, N) containing the covariance matrices of each class in the training set.
                          # HINT: USE NP.COV (Pay attenention for what it takes as an argument)

for classIndex in range(numClasses):
    # TODO [5]: Estimate the parameters of the Gaussian distributions of the given classes.
    # Fill pClasses, estimate_means, and estimate_covariances in this part 
    # Your code should be vectorized WITHOUT USING A SINGLE FOR LOOP.
    pClasses[classIndex] = np.count_nonzero(ytrain == (classIndex+1))
    mask = (Xtrain[:, 0] == classIndex+1)
    estimate_means[classIndex]=np.mean(Xtrain[mask, 1:],axis=0)
    estimate_covariances[classIndex]=np.cov(Xtrain[mask, 1:],rowvar=False)
pClasses=pClasses/(len(ytrain))
estimate_means = np.array(estimate_means)
estimate_covariances = np.array(estimate_covariances)



predicted_classes = np.zeros((K,1)) # predicted_classes: A numpy array of size (K, 1) where K is the number of points in the test set. Every element in this array
                                    # contains the predicted class of Bayes classifier for this test point.

for i in range(Xtest.shape[0]):
    classProbabilities = np.zeros(numClasses)
    # TODO [7.A]: Compute the probability that the test point X_Test[i] belongs to each class in numClasses.
    #  Fill the array classProbabilities accordingly.
    for j in range(3):    
        classProbabilities[j] = multivariate_normal_gaussian(Xtest[i],estimate_means[j],estimate_covariances[j])*pClasses[j]

    # TODO [7.B]: Find the prediction of the test point X_Test[i] and append it to the predicted_classes array.
    predicted_classes[i]=np.argmax(classProbabilities)+1
    


accuracy = np.count_nonzero(  predicted_classes == ytest  ) / len(ytest)
print('Accuracy = ' + accuracy )