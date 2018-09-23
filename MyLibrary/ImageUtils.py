import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import statistics
import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

MLPATH = os.path.dirname(os.path.realpath(__file__)).split('\\')
MLPATH = '\\'.join(MLPATH[:len(MLPATH) - 1]) + "\\MLParam\\"


def network01E(x, y, test=False):
    # Input:x -> 1,64,48
    # BinaryConnectConvolution -> 64,60,44
    h = PF.binary_connect_convolution(x, 64, (5,5), (0,0), name='BinaryConnectConvolution')
    # MaxPooling -> 64,30,22
    h = F.max_pooling(h, (2,2), (2,2))
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.5, 0.01, not test, name='BatchNormalization')
    # BinarySigmoid
    h = F.binary_sigmoid(h)
    # BinaryConnectConvolution_2 -> 64,26,18
    h = PF.binary_connect_convolution(h, 64, (5,5), (0,0), name='BinaryConnectConvolution_2')
    # MaxPooling_2 -> 64,13,9
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
nn.load_parameters(MLPATH + "./01E.h5")

# Prepare input variable
var01E = nn.Variable((1,1,64,48))

# Let input data to x.d
# x.d = ...
var01E.data.zero()

# Build network for inference
ans01E = network01E(var01E, None, test=True)

def toGreyImg(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarize(img):
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
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

def ocr01E(img):
	img = img.copy()
	img = denoiseMedian(img)
	img = binarize(img)

	img = cv2.resize(img, (48, 64), interpolation=cv2.INTER_CUBIC)
	img = img / 255

	var01E.d = img

	ans01E.forward()
	return ans01E.d.argmax(axis=1)[0]

def avgGreyVal(img):
	img = denoiseMedian(img)
	img = binarize(np.copy(img))
	return np.mean(img)

def mser(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = denoiseGaussian(img)
	ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
	h, w = img.shape
	rects = []

	mser = cv2.MSER_create()
	regions, _ = mser.detectRegions(img)
	for p in regions:
		xmax, ymax = np.amax(p, axis=0)
		xmin, ymin = np.amin(p, axis=0)

		xDiff = abs(xmax - xmin)
		yDiff = abs(ymax - ymin)
		
		isPortrait = xDiff < yDiff
		isGoodHeight = (yDiff > (0.01 * h)) and (yDiff < (0.1 * h)) 
		isGoodWidth = (xDiff > (0.005 * w)) and (xDiff < (0.1 * w))
		isGoodRatio = ((xDiff / yDiff) < 0.75)

		if isPortrait and isGoodHeight and isGoodWidth:
			rects.append((xmin, ymin, xmax, ymax))
	return rects

def filterInvalidTextRegions(img, regions):
	h, w, d = img.shape
	done = []
	for i in range(len(regions)):
		done.append(False)
	noRepetition = []

	for i in range(len(regions)):
		current = regions[i]
		if not done[i]:
			done[i] = True
			xMaxList = [current[2]]
			xMinList = [current[0]]
			yMaxList = [current[3]]
			yMinList = [current[1]]
			for j in range(len(regions)):
				if not done[j]:
					target = regions[j]
					center = (int(round(np.mean([target[0], target[2]]))), int(round(np.mean([target[1], target[3]]))))
					if (center[0] in range(current[0], current[2])) and \
						(center[1] in range(current[1], current[3])):
						done[j] = True
						xMinList.append(target[0])
						yMinList.append(target[1])
						xMaxList.append(target[2])
						yMaxList.append(target[3])
			
			xMin = int(round(statistics.median(xMinList)))
			yMin = int(round(statistics.median(yMinList)))
			xMax = int(round(statistics.median(xMaxList)))
			yMax = int(round(statistics.median(yMaxList)))
			noRepetition.append((xMin, yMin, xMax, yMax))

	withWhiteBorder = []
	AMP = 1.75
	HSIDERATIO = 0.75
	MARGINWIDTH = 3
	for r in noRepetition:
		center = (np.mean([r[0], r[2]]), np.mean([r[1], r[3]]))
		xDiff = abs(r[2] - r[0])
		yDiff = abs(r[3] - r[1])

		yShift = (AMP * yDiff) / 2
		xShift = HSIDERATIO * yShift

		xMin = max(int(round(center[0] - xShift)), 0)
		yMin = max(int(round(center[1] - yShift)), 0)
		xMax = min(int(round(center[0] + xShift)), w)
		yMax = min(int(round(center[1] + yShift)), h)

		testScanImg = np.copy(img[yMin:yMax, xMin:xMax])
		reducedHeight, reducedWidth, reducedDepth = testScanImg.shape
		testScanImg[MARGINWIDTH:reducedHeight-(MARGINWIDTH + 1), \
		MARGINWIDTH:reducedWidth-(MARGINWIDTH + 1)] = (255, 255, 255) #white out everything except the outer lining
		if avgGreyVal(testScanImg) == 255:
			withWhiteBorder.append((xMin, yMin, xMax, yMax))
		# else:
			# print("REJECTED: Greyval = {}".format(avgGreyVal(testScanImg)))
	
	return withWhiteBorder

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