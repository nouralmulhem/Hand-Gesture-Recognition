import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern



def lbp(img, radius=3, n_points=8):
    """
    Compute Local Binary Pattern (LBP) features for an image.

    Parameters:
        img: 2D numpy array representing the image
        radius: radius of the circular LBP sampling region (default: 3)
        n_points: number of sampling points in the circular region (default: 8)

    Returns:
        2D numpy array representing the LBP features
    """
    lbp_img = local_binary_pattern(img, n_points, radius, 'uniform')
    hist, _ = np.histogram(lbp_img.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Compute Histogram of Oriented Gradients (HOG) features for an image.

    Parameters:
        img: 2D numpy array representing the image
        orientations: number of gradient orientation bins (default: 9)
        pixels_per_cell: size of a cell in pixels (default: (8, 8))
        cells_per_block: number of cells in each block (default: (3, 3))

    Returns:
        1D numpy array representing the HOG features
    """
    hog_feats,img = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                    visualize=True, feature_vector=True)
    return hog_feats,img



def shiThomasFeatureExtraction(grayImage,noOfCorners,qualityLevel,distance):
    # Quality level => between 0-1 which denotes the minimum quality of corner below which everyone is rejected
    # distance => minimum euclidean distance between corners detected.

    # With all this information, the function finds corners in the image. All
    # corners below quality level are rejected. Then it sorts the remaining 
    # corners based on quality in the descending order. Then function takes
    # first strongest corner, throws away all the nearby corners in the range
    # of minimum distance and returns N strongest corners.
    corners = cv2.goodFeaturesToTrack(grayImage, noOfCorners, qualityLevel, distance)
    # corners = np.int0(corners)
    
    # print("coreners before",corners)
    # corners = corners.reshape(corners.shape[0], 2)
    corners = corners.ravel()
    # print("coreners after",corners)

    # corner_values=[]
    # if(len(corners)>0):
    #     corner_values = [grayImage[int(y), int(x)] for x, y in corners[:, 0]]
    # for i in corners:
    #     x, y = i.ravel()
    #     corner_values.append(grayImage[int(y), int(x)])
    #  cv.circle(img, (x, y), 3, [255, 255, 0], -1)
    #  cv.imshow('Shi-Tomasi Corner Detector', img)
    # corner_values=np.array(corner_values, dtype=np.float32)
    return corners
  
# path = "./Dataset/men/3/3_men (10).JPG"
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (128, 64))
# cv2.imshow("Image", img)

# LBP_features = lbp(img, radius=3, n_points=8)
# print(LBP_features)

# HOG_features,Hog_img = hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
# print(HOG_features)
# cv2.imshow("HOG", Hog_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def ORB_features(img):
    
    orb = cv2.ORB_create(nfeatures=10)
    keypoints_orb, descriptors = orb.detectAndCompute(img, None)

    return keypoints_orb, descriptors

# img = cv2.drawKeypoints(img, keypoints_orb, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def SIFT_features(img):
    
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors = sift.detectAndCompute(img, None)

    return keypoints_orb, descriptors
