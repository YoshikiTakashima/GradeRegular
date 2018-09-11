import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
from PIL import ImageFilter
import pytesseract

def toGreyImg(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarize(img):
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
	return thresh

def denoiseGaussian(img):
	return(cv2.GaussianBlur(img,(5,5),0))

def denoiseMedian(img):
	return cv2.medianBlur(img,5)

def distanceBetween(xy1, xy2):
	return math.hypot((xy1[0] - xy2[0]), (xy1[1] - xy2[1]))

def show(img):
	plt.imshow(img, cmap = 'gray', interpolation = None)
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()

def tesseractOCR(img):
	answer = ''
	ocrImg = Image.fromarray(img)
	answer = pytesseract.image_to_string(ocrImg)
	return answer

def main():
	#from sys import argv
	img = cv2.imread('./Examples/handDFA.jpg')

	img = toGreyImg(img)
	#img = denoise(img)
	img = binarize(img)

	cv2.imshow('Binarized',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()