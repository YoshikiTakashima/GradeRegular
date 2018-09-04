import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
from PIL import ImageFilter
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import cv2
import numpy

def network(x, y, test=False):
	# Input:x -> 1,28,28
	# BinaryConnectConvolution -> 64,24,24
	h = PF.binary_connect_convolution(x, 64, (5,5), (0,0), name='BinaryConnectConvolution')
	# MaxPooling -> 64,12,12
	h = F.max_pooling(h, (2,2), (2,2))
	# BatchNormalization
	h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization')
	# BinarySigmoid
	h = F.binary_sigmoid(h)
	# BinaryConnectConvolution_2 -> 64,8,8
	h = PF.binary_connect_convolution(h, 64, (5,5), (0,0), name='BinaryConnectConvolution_2')
	# MaxPooling_2 -> 64,4,4
	h = F.max_pooling(h, (2,2), (2,2))
	# BatchNormalization_2
	h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization_2')
	# BinarySigmoid_2
	h = F.binary_sigmoid(h)
	# BinaryConnectAffine -> 512
	h = PF.binary_connect_affine(h, (512,), name='BinaryConnectAffine')
	# BatchNormalization_3
	h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization_3')
	# BinarySigmoid_3
	h = F.binary_sigmoid(h)
	# BinaryConnectAffine_2 -> 10
	h = PF.binary_connect_affine(h, (10,), name='BinaryConnectAffine_2')
	# BatchNormalization_4
	h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization_4')
	# Softmax
	h = F.softmax(h)
	# CategoricalCrossEntropy -> 1
	# h = F.categorical_cross_entropy(h, y) #Cannot be used.
	return h

# load parameters
nn.clear_parameters()
nn.load_parameters("../MLParam/parameters.h5")

# Prepare input variable
x=nn.Variable((1,1,28,28))
x.data.zero()

# Build network for inference
y = network(x, "", test=True)

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

def ocrToString(img):
	mapList = ['n', '0', '1', 'E', 'f', 's']
	# Let input data to x.d
	# x.d = ...
	img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC) 
	img = cv2.resize(img, (75, 75), interpolation=cv2.INTER_CUBIC) 
	img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
	ret,img = cv2.threshold(img,230,255,cv2.THRESH_BINARY) 

	show(img)
	img = img / 255

	x.d = (img)

	# Execute inference
	y.forward()
	
	return mapList[y.d.argmax(axis=1)[0]]


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