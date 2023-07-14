import pandas as pd
from skimage.color import rgb2gray
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv2
import skimage.io as io
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
from skimage.filters import gaussian
from tempfile import TemporaryFile
import numpy as np
from matplotlib.pyplot import imshow
from CommunFunction import *
from PIL import Image
import PIL
import os
import glob


def resize(image, width=200, hsize=None):
    # Function to resize an image

    base_width = width

    # Calculate the width percentage to maintain the aspect ratio
    width_percent = (base_width / float(image.size[0]))
    # If hsize is not provided, calculate the height based on the width percentage
    hsize = int(
        (float(image.size[1]) * float(width_percent))) if hsize == None else hsize
    # Resize the image using the calculated width and height
    image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
    return image


def adjust_gamma(image, gamma=1.0):
    # Function to adjust gamma correction of an image

    # Calculate the inverse of the gamma value
    invGamma = 1.0 / gamma
    # Create a lookup table for gamma correction
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def show_histo(img, name=''):
    # Display the histogram of a grayscale image.

    dst = cv2.calcHist(img, [0], None, [256], [0, 256])
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale image'+name)
    plt.show()


def get_result(img, mask):
    # Function to get the result image with a given mask

    copy = img.copy()
    channels = img.shape[2]
    # Apply bitwise AND operation between each channel of the image and the mask
    for i in range(channels):
        copy[:, :, i] = np.bitwise_and(copy[:, :, i], mask)
    temp = mask.copy()
    # Convert 0 values in the mask to 255 (white) and 255 values to 0 (black)
    temp[mask == 0] = 255
    temp[mask == 255] = 0
    # Apply bitwise OR operation between each channel of the image and the modified mask
    for i in range(channels):
        copy[:, :, i] = np.bitwise_or(copy[:, :, i], temp)
    return copy


def shadow_remove(img):
    # Function to remove shadow from an image

    rgb_planes = cv2.split(img)
    result_norm_planes = []
    # Iterate over each color plane in the image
    for plane in rgb_planes:
        # Dilate the plane using a 7x7 kernel
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))

        # Apply median blur with a kernel size of 21x21 to obtain the background image
        bg_img = cv2.medianBlur(dilated_img, 21)

        # Compute the absolute difference between the plane and the background image
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        # Normalize the difference image to the range of 0-255
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Append the normalized image to the result list
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


def reduce_image(img, thickness):
    # Function to reduce the size of an image

    mask1 = img.copy()

    # Set the top thickness rows to 0 (black)
    mask1[:thickness, :] = 0

    # Set the bottom thickness rows to 0 (black)
    mask1[-1*thickness:, :] = 0

    # Set the left thickness columns to 0 (black)
    mask1[:, :thickness] = 0

    # Set the right thickness columns to 0 (black)
    mask1[:, -1*thickness:] = 0
    return mask1


def get_binary_low_medium_constract(img_RGB):
    # Function to get the binary image with low to medium contrast

    gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)

    # Thresholding the saturation channel of the HSV image
    ret, img_binary = cv2.threshold(
        img_HSV[:, :, 1], 50, 255, cv2.THRESH_BINARY)

    # Thresholding the value channel of the HSV image
    ret, img_binary2 = cv2.threshold(
        img_HSV[:, :, 2], 50, 255, cv2.THRESH_BINARY)

    # Combining the two binary images
    img_binary[img_binary2 == 0] = 0

    # Create a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply erosion to the image
    erode_img = cv2.erode(img_binary, kernel, iterations=2)

    # Apply dilation to the eroded image
    dilate_img = cv2.dilate(erode_img, kernel, iterations=2)
    return dilate_img


def get_binary_heigh_constract(img_RGB):
    # Function to get the binary image with high contrast

    # img_RGBA = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2RGBA)
    # img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    # ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    # img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    # img_YCC = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCR_CB)
    # img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)

    # Convert RGB image to YUV color space
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)

    # Make float and divide by 255 to give BGRdash
    # bgrdash = img_BGR.astype(np.float)/255.

    # hls_img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    # lab_img = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)

    # img_HSV[:, :, 0] = cv2.equalizeHist(img_HSV[:, :, 0])
    # img_HSV[:, :, 1] = cv2.equalizeHist(img_HSV[:, :, 1])
    # img_HSV[:, :, 2] = cv2.equalizeHist(img_HSV[:, :, 2])

    # img_YCC[:, :, 0] = cv2.equalizeHist(img_YCC[:, :, 0])
    # img_YCC[:, :, 1] = cv2.equalizeHist(img_YCC[:, :, 1])
    # img_YCC[:, :, 2] = cv2.equalizeHist(img_YCC[:, :, 2])

#     img_YUV[:, :, 0] = cv2.equalizeHist(img_YUV[:, :, 0])
#     img_YUV[:, :, 1] = cv2.equalizeHist(img_YUV[:, :, 1])

    # Equalize the Y channel of the YUV image
    img_YUV[:, :, 2] = cv2.equalizeHist(img_YUV[:, :, 2])

    # hls_img[:, :, 0] = cv2.equalizeHist(hls_img[:, :, 0])
    # hls_img[:, :, 1] = cv2.equalizeHist(hls_img[:, :, 1])
    # hls_img[:, :, 2] = cv2.equalizeHist(hls_img[:, :, 2])

    # lab_img[:, :, 0] = cv2.equalizeHist(lab_img[:, :, 0])
    # lab_img[:, :, 1] = cv2.equalizeHist(lab_img[:, :, 1])
    # lab_img[:, :, 2] = cv2.equalizeHist(lab_img[:, :, 2])

    # Apply thresholding to obtain a binary image
    ret, img_binary1 = cv2.threshold(
        img_YUV[:, :, 2], 150, 255, cv2.THRESH_BINARY)

    # Create a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform erosion and dilation on the binary image
    dilate_img = cv2.erode(img_binary1, kernel, iterations=1)
    erode_img = cv2.dilate(dilate_img, kernel, iterations=1)

    # Return the binary image with high contrast
    return erode_img


def calculate_brightness(gray, index=None):
    # Function to calculate the brightness of an image

    # Calculate the mean brightness value of the grayscale image
    mean_value = cv2.mean(gray)[0]

    # If an index is provided, display the grayscale image and print the mean value
    if index is not None:
        show_images([gray], [str(index)])
        print("the image "+str(index)+" has mean value : ", mean_value)

    # Return the mean brightness value
    return mean_value


def image_pre_processing(image):
    # Function for image preprocessing
    # it is the main function in the image processing

    # Resize the image to the desired dimensions
    image = resize(image, 4 * 128, 4 * 64)

    # Convert the resized image to RGB and grayscale
    img_RGB = np.array(image)
    gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

    # Calculate the mean brightness value of the grayscale image
    mean_value = calculate_brightness(gray)
    dilate_img = None

    # Determine the appropriate image processing based on the mean brightness value
    if (mean_value >= 190):
        dilate_img = get_binary_heigh_constract(img_RGB)
    else:
        dilate_img = get_binary_low_medium_constract(img_RGB)

    # Convert the dilated image to binary (0 or 1) and create a mask
    dilate_img[dilate_img == 255] = 1
    mask1 = dilate_img
    mask1[mask1 != 1] = 0
    mask1[mask1 == 1] = 255

    # cut the border of the image
    mask1 = reduce_image(img=mask1, thickness=3)

    # Apply Gaussian blur and Canny edge detection to the mask
    blured = cv2.GaussianBlur(mask1, (5, 5), 0)
    edges = cv2.Canny(blured, threshold1=40, threshold2=100,
                      apertureSize=3, L2gradient=False)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Close the edges using morphological closing
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed edges
    contours, hierarchy = cv2.findContours(
        closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Create a mask and find the largest contour
    mask = np.zeros_like(edges)
    max_area = 0
    second_max = 0
    biggest_contour = None
    second_biggest = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            biggest_contour = cnt
    # If there is at least one contour, draw it on the mask
    if len(contours) > 0:
        mask = cv2.drawContours(
            mask, [biggest_contour], -1, (255, 255, 255), -1)

    # Generate the result by applying the mask to the original image
    result = get_result(img_RGB, mask)

    # Apply a 5x5 averaging kernel to the result
    kernel = np.ones((5, 5), np.float32)/25
    result = cv2.filter2D(result, -1, kernel)
    if len(contours) > 0:
        (x, y, w, h) = cv2.boundingRect(biggest_contour)
        result = result[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]
        print(x, y, w, h)

    # Resize the result to the desired dimensions and convert to grayscale
    result = cv2.resize(result, (128, 64))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
#     show_images([ result, rotated_img])
    # Return the mask and preprocessed result image as a tuple
    return mask, result


def process_image_thread(image, index, results, binaries):
    # Process an image and store the results in the provided lists.

    # Perform image pre-processing to obtain the binary mask and processed image
    binary, result = image_pre_processing(image)

    # Store the processed image and binary mask in the corresponding lists at the given index
    results[index] = result
    binaries[index] = binary
    return
