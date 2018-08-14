import cv2 as cv
import numpy as np

def binarize(img):
    ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    return 0