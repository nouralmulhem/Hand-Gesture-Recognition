import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from utils import tunning_feature_extraction
# Load positive and negative image samples
y,pos_images = tunning_feature_extraction("./Dataset/")
_,neg_images = tunning_feature_extraction("./Dataset/")

# Define Haar-like feature parameters
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
n_bins = 9

# Initialize HOG descriptor
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

# Compute Haar-like features for positive images
pos_features = []
for image in pos_images:
    img = cv2.resize(image, win_size)
    feature = hog.compute(img)
    pos_features.append(feature)

# Compute Haar-like features for negative images
neg_features = []
for image in neg_images:
    img = cv2.resize(image, win_size)
    feature = hog.compute(img)
    neg_features.append(feature)

# Create training dataset
X = np.vstack((pos_features, neg_features))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train SVM model
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Evaluate model on test set
accuracy = clf.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
