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

path = "men/3/3_men (10).JPG"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 64))
cv2.imshow("Image", img)

LBP_features = lbp(img, radius=3, n_points=8)
print(LBP_features)

HOG_features,Hog_img = hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
print(len(HOG_features))
cv2.imshow("HOG", Hog_img)
cv2.waitKey(0)
cv2.destroyAllWindows()