import cv2
import numpy as np

# Load the image
image = cv2.imread('./Dataset/men/0/0_men (1).JPG')

# Convert the image to the Lab color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Define the lower and upper bounds for the skin color range in Lab color space
lower_skin = np.array([0, 20, 77], dtype=np.uint8)
upper_skin = np.array([255, 135, 127], dtype=np.uint8)

# Create a mask based on the skin color range
mask = cv2.inRange(lab, lower_skin, upper_skin)

# Perform morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Hand', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
