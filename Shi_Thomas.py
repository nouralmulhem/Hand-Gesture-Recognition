import numpy as np
import cv2 as cv

def shiThomasFeatureExtraction(grayImage,noOfCorners,qualityLevel,distance):
   # Quality level => between 0-1 which denotes the minimum quality of corner below which everyone is rejected
   # distance => minimum euclidean distance between corners detected.


   # With all this information, the function finds corners in the image. All
   # corners below quality level are rejected. Then it sorts the remaining 
   # corners based on quality in the descending order. Then function takes
   # first strongest corner, throws away all the nearby corners in the range
   # of minimum distance and returns N strongest corners.

   corners = cv.goodFeaturesToTrack(grayImage, noOfCorners, qualityLevel, distance)
   corners = np.int0(corners)
   # for i in corners:
   #  x, y = i.ravel()
   #  cv.circle(img, (x, y), 3, [255, 255, 0], -1)
   #  cv.imshow('Shi-Tomasi Corner Detector', img)
   return corners.reshape(corners.shape[0]*corners.shape[1]*corners.shape[2])
  

# path = "./Dataset/men/3/3_men (10).JPG"
# img = cv.imread(path, cv.IMREAD_GRAYSCALE)
# img = cv.resize(img, (128, 64))

# features = shiThomasFeatureExtraction(img,100,0.01,10)
# print(features.reshape(features.shape[0]*features.shape[1]*features.shape[2]))
