import pandas as pd
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv2
import cv2 as cv
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
from PIL import Image


def resize(image, width=200, hsize=None):
    base_width = width
    width_percent = (base_width / float(image.size[0]))
    hsize = int(
        (float(image.size[1]) * float(width_percent))) if hsize == None else hsize
    image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
    return image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    #     their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def show_histo(img):
    dst = cv2.calcHist(img, [0], None, [256], [0, 256])

    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale image')
    plt.show()


def get_result(img, mask):
    copy = img.copy()
    channels = img.shape[2]
    for i in range(channels):
        copy[:, :, i] = np.bitwise_and(copy[:, :, i], mask)
    return copy


def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


def reduce_image(img, thickness):
    mask1 = img.copy()
    mask1[:thickness, :] = 0
    mask1[-1*thickness:, :] = 0
    mask1[:, :thickness] = 0
    mask1[:, -1*thickness:] = 0
    return mask1


def get_binary_lowhigth_constract(img_RGB):
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_HSV[:, :, 0] = cv2.equalizeHist(img_HSV[:, :, 0])
    img_HSV[:, :, 1] = cv2.equalizeHist(img_HSV[:, :, 1])
    img_HSV[:, :, 2] = cv2.equalizeHist(img_HSV[:, :, 2])
    ret, img_binary = cv2.threshold(
        img_HSV[:, :, 1], 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    # Apply erosion to the image
    erode_img = cv2.erode(img_binary, kernel, iterations=2)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=2)
    return dilate_img


def image_pre_processing(image):
    image = resize(image, 4 * 128, 4 * 64)
    img_RGB = np.array(image)
    img_RGBA = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2RGBA)
    img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_YCC = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCR_CB)
    img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)

    # Make float and divide by 255 to give BGRdash
    bgrdash = img_BGR.astype(np.float)/255.

    # # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
    # K = 1 - np.max(bgrdash, axis=2)

    # # Calculate C
    # C = (1-bgrdash[..., 2] - K)/(1-K)

    # # Calculate M
    # M = (1-bgrdash[..., 1] - K)/(1-K)

    # # Calculate Y
    # Y = (1-bgrdash[..., 0] - K)/(1-K)

    # # Combine 4 channels into single image and re-scale back up to uint8
    # CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    hls_img = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    lab_img = cv.cvtColor(img_RGB, cv.COLOR_RGB2LAB)

    img_HSV_copy = img_HSV.copy()
    img_YCC_copy = img_YCC.copy()
    img_YUV_copy = img_YUV.copy()
    # CMYK_copy = CMYK.copy()
    hls_img_copy = hls_img.copy()

    img_HSV[:, :, 0] = cv2.equalizeHist(img_HSV[:, :, 0])
    img_HSV[:, :, 1] = cv2.equalizeHist(img_HSV[:, :, 1])
    img_HSV[:, :, 2] = cv2.equalizeHist(img_HSV[:, :, 2])

    img_YCC[:, :, 0] = cv2.equalizeHist(img_YCC[:, :, 0])
    img_YCC[:, :, 1] = cv2.equalizeHist(img_YCC[:, :, 1])
    img_YCC[:, :, 2] = cv2.equalizeHist(img_YCC[:, :, 2])

    img_YUV[:, :, 0] = cv2.equalizeHist(img_YUV[:, :, 0])
    img_YUV[:, :, 1] = cv2.equalizeHist(img_YUV[:, :, 1])
    img_YUV[:, :, 2] = cv2.equalizeHist(img_YUV[:, :, 2])

    # CMYK[:, :, 0] = cv2.equalizeHist(CMYK[:, :, 0])
    # CMYK[:, :, 1] = cv2.equalizeHist(CMYK[:, :, 1])
    # CMYK[:, :, 2] = cv2.equalizeHist(CMYK[:, :, 2])

    hls_img[:, :, 0] = cv2.equalizeHist(hls_img[:, :, 0])
    hls_img[:, :, 1] = cv2.equalizeHist(hls_img[:, :, 1])
    hls_img[:, :, 2] = cv2.equalizeHist(hls_img[:, :, 2])

    lab_img[:, :, 0] = cv2.equalizeHist(lab_img[:, :, 0])
    lab_img[:, :, 1] = cv2.equalizeHist(lab_img[:, :, 1])
    lab_img[:, :, 2] = cv2.equalizeHist(lab_img[:, :, 2])

#     show_images([img_HSV,img_YCC,CMYK,hls_img,lab_img,img_YUV] , ['img_HSV','img_YCC','CMYK','hls_img','lab_img','img_YUV'])
#     show_images([img_YUV[:,:,0],img_YUV[:,:,1],img_YUV[:,:,2]])

    ret, img_binary1 = cv2.threshold(
        img_YUV[:, :, 2], 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilate_img = cv2.dilate(img_binary1, kernel, iterations=1)
    erode_img = cv2.erode(dilate_img, kernel, iterations=1)

    mask1 = dilate_img
    mask1 = get_binary_lowhigth_constract(img_RGB)
    mask1 = reduce_image(img=mask1, thickness=3)

    blured = cv2.GaussianBlur(mask1, (5, 5), 0)
    edges = cv2.Canny(blured, threshold1=40, threshold2=100,
                      apertureSize=3, L2gradient=False)

    # Define the kernel for closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Perform closing operation
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # show_images([img_gray,edges,closed_edges],['img_gray','edges','closed_edges'])
    contours, hierarchy = cv2.findContours(
        closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    mask = np.zeros_like(edges)
    max_area = 0
    second_max = 0
    biggest_contour = None
    second_biggest = None
    for cnt in contours:
        #     mask=cv2.drawContours(mask, [cnt], -1, (255,255,255), -1)
        area = cv2.contourArea(cnt)
    #     print(area)
        if area > max_area:
            second_max = max_area
            second_biggest = biggest_contour
            max_area = area
            biggest_contour = cnt
        elif area > second_max:
            second_max = area
            second_biggest = cnt
    # print(len(cnt))
    mask = cv2.drawContours(mask, [biggest_contour], -1, (255, 255, 255), -1)
    # print(biggest_contour)
    result = get_result(img_RGB, mask)
#     show_images([img_RGB,mask,result],['origenal image','masked image','final'])
    return mask, result


# image = Image.open("./Dataset/men/1/1_men (120).jpg").convert('RGB')
# img = np.array(image)
# binary, result = image_pre_processing(image)
# show_images([img, binary, result], ['origenal', 'binary ', 'result'])
