import cv2
import numpy as np
path = "Dataset_0-5/img2.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (4*128, 4*64))
cv2.imshow("Image", img)


sift = cv2.SIFT_create()
# surf = cv2.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

keypoints_sift, descriptors = sift.detectAndCompute(img, None)
# keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)
print(len(keypoints_sift))
print(len(keypoints_orb))

# img = cv2.drawKeypoints(img, keypoints_sift, None)
# img = cv2.drawKeypoints(img, keypoints_surf, None)
img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()