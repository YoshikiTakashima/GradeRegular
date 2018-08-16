import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def toGreyImg(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def binarize(img):
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
    return thresh

def denoise(img):
    return(cv.GaussianBlur(img,(5,5),0))

def show(img):
    cv.imshow('Lines',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    #from sys import argv
    img = cv.imread('./Examples/UtilTest/handDFA.jpg')

    img = toGreyImg(img)
    #img = denoise(img)
    img = binarize(img)

    cv.imshow('Binarized',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()