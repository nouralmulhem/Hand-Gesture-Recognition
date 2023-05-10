from imutils.perspective import four_point_transform
from matplotlib import pyplot as plt
from imutils import contours
import numpy as np
import argparse
import os
import imutils
import cv2


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def getEdges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged


def getEdgesTable(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blurred, 30, 100)
    return edged


def getContours(edged):
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    boarderd = []
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            points = len(approx)
            if (points == 8):
                docCnt = approx
                break
    return docCnt


# In[ ]:


def CropImg(img):
    cropped_image = img[int(img.shape[0] * .01):int(img.shape[0] * .98),
                        int(img.shape[1] * .02):int(img.shape[1] * .98)]
    return cropped_image


def exetractTable(image, docCnt):
    Table = four_point_transform(image, docCnt.reshape(4, 2))
    return Table


def RemoveText(image):
    kernel = np.ones((8, 8), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    return img


def widerTable(docCnt):
    widerdoc = np.copy(docCnt)
    temparr = []
    for c in widerdoc:
        temparr.append(c[0])
    temparr.sort(key=lambda temparr: temparr[0])
    temparr[0][0] = temparr[0][0] - 20
    temparr[1][0] = temparr[1][0] - 20

    temparr[2][0] = temparr[2][0] + 20
    temparr[3][0] = temparr[3][0] + 20

    for i in (0, 1, 2, 3):
        widerdoc[i][0] = temparr[i]

    return widerdoc


def GetTagbleOfImagesInDir(folder_dir):
    for image_name in os.listdir(folder_dir):
        print("Start : "+folder_dir+image_name)
        image = cv2.imread(folder_dir+image_name, 1)
        Paper = getPaper(image)
        Table = getTable(Paper, image)
        Save_img(Table, image_name)
        print("Done : "+folder_dir+image_name)


def GetCellsOfImagesInDir(folder_dir):
    for image_name in os.listdir(folder_dir):
        print("Start : "+folder_dir+image_name)
        image = cv2.imread(folder_dir+image_name, 1)
        Paper = getPaper(image)
        Table = getTable(Paper, image)
        Cells = get_cells(Table)
        Save_img(Cells, image_name)
        print("Done : "+folder_dir+image_name)


def getPaper(image):
    edged = getEdges(image)
    kernel = np.ones((15, 15), np.uint8)
    img_dilation = cv2.dilate(edged, kernel, iterations=1)
    docCnt = getContours(img_dilation)
    Table = exetractTable(image, docCnt)
    return Table


def getTable(Paper, image):
    Table = Paper
    CropPaper = CropImg(Paper)
    imageArea, PaperArea = image.shape[0] * \
        image.shape[1], Paper.shape[0] * Paper.shape[1]
    if (imageArea < PaperArea * 2):
        edged = getEdgesTable(CropPaper)
        kernel2 = np.ones((5, 10), np.uint8)
        img_dilation = cv2.dilate(edged, kernel2, iterations=1)
        docCnt = getContours(img_dilation)
        Table = exetractTable(CropPaper, docCnt)
    return Table


def get_cells(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, 127, 255, cv2.THRESH_BINARY)
    img_bin = ~img_bin

    line_min_width = int(image.shape[1] * .5)

    line_min_heigth = image.shape[0]

    kernal_h = np.ones((1, line_min_width), np.uint8)
    img_bin_h = cv2.erode(img_bin, kernal_h, iterations=1)
    kernal_h_diolation = np.ones((1, image.shape[1]), np.uint8)
    img_bin_h_after = cv2.dilate(img_bin_h, kernal_h_diolation, iterations=1)

    show_images([img_bin, img_bin_h, img_bin_h_after], [
                "img_bin", "img_bin_h", "img_bin_h_after"])

    return img_bin_h


def Showedged(folder_dir):
    for image_name in os.listdir(folder_dir):
        image = cv2.imread(folder_dir+image_name, 1)
        Paper = getPaper(image)
        Table = getTable(Paper, image)
        Cells = get_cells(Table)
        show_images([image, Paper, Table, Cells, img_bin], [
                    "image", "Paper", "Table", "Cells", "img_bin"])


def Save_img(image, Name):
    cv2.imwrite(os.path.join("./Cells", Name), image)
