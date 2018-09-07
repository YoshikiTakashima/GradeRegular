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
    # Input:x -> 1,64,64
    # BinaryConnectConvolution -> 64,60,60
    h = PF.binary_connect_convolution(x, 64, (5,5), (0,0), name='BinaryConnectConvolution')
    # MaxPooling -> 64,30,30
    h = F.max_pooling(h, (2,2), (2,2))
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization')
    # BinarySigmoid
    h = F.binary_sigmoid(h)
    # BinaryConnectConvolution_2 -> 64,26,26
    h = PF.binary_connect_convolution(h, 64, (5,5), (0,0), name='BinaryConnectConvolution_2')
    # MaxPooling_2 -> 64,13,13
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
    # h = F.categorical_cross_entropy(h, y)
    return h

# load parameters
nn.clear_parameters()
nn.load_parameters("../MLParam/parameters.h5")

# Prepare input variable
mlVar=nn.Variable((1,1,64,64))
mlVar.data.zero()

# Build network for inference
mlAnswer = network(mlVar, "", test=True)

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

STEP = 0.3
SCALE = 0.6
def ocrToString(img):
	mapList = [ '0', '1', 'E', 'f', 'n', 's']
	h, w = img.shape
	detectList = [] 
	side = min(h, w)
	while side >= 64:
		microstep = int(round(side * STEP))
		for x in range(0, w - side - 1, microstep):
			for y in range(0, h - side - 1, microstep):
				# print("x: {} // y: {} // side: {}".format(x, y, side))
				current = np.copy(img[x:x + side, y:y + side])
				# show(current)
				current = cv2.resize(current, (64, 64), interpolation=cv2.INTER_CUBIC)
				current = cv2.medianBlur(current,3)
				ret,current = cv2.threshold(current, 240, 255, cv2.THRESH_BINARY)
				current = current / 255

				mlVar.d = (current)
				mlAnswer.forward()
				if mapList[mlAnswer.d.argmax(axis=1)[0]] != 'n':
					print("FOUND: {}".format(mapList[mlAnswer.d.argmax(axis=1)[0]]))
					show(img[x:x + side, y:y + side])
					img[x:x + side, y:y + side] = 255
					detectList.append(mapList[mlAnswer.d.argmax(axis=1)[0]])
		side = int(round(SCALE * side))
	
	return detectList


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