import cv2
import numpy as np
import math
import statistics
from matplotlib import pyplot as plt
import ImageUtils

import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

MLPATH = os.path.dirname(os.path.realpath(__file__)).split('\\')
MLPATH = '\\'.join(MLPATH[:len(MLPATH) - 1]) + "\\MLParam\\"

def binaryTargeting(x, y, test=False):
    # Input:x -> 1,64,64
    # Convolution -> 16,60,60
    h = PF.convolution(x, 16, (5,5), (0,0), name='Convolution')
    # MaxPooling -> 16,30,30
    h = F.max_pooling(h, (2,2), (2,2))
    # Tanh
    h = F.tanh(h)
    # Convolution_2 -> 8,26,26
    h = PF.convolution(h, 8, (5,5), (0,0), name='Convolution_2')
    # MaxPooling_2 -> 8,13,13
    h = F.max_pooling(h, (2,2), (2,2))
    # Tanh_2
    h = F.tanh(h)
    # Affine -> 10
    h = PF.affine(h, (10,), name='Affine')
    # Tanh_3
    h = F.tanh(h)
    # Affine_2 -> 1
    h = PF.affine(h, (1,), name='Affine_2')
    # Sigmoid
    h = F.sigmoid(h)
    # BinaryCrossEntropy
    # h = F.binary_cross_entropy(h, y)
    return h

def letterOrNum(x, y, test=False):
    # Input:x -> 1,64,64
    # Convolution -> 16,60,60
    h = PF.convolution(x, 16, (5,5), (0,0), name='Convolution')
    # MaxPooling -> 16,30,30
    h = F.max_pooling(h, (2,2), (2,2))
    # Tanh
    h = F.tanh(h)
    # Convolution_2 -> 8,26,26
    h = PF.convolution(h, 8, (5,5), (0,0), name='Convolution_2')
    # MaxPooling_2 -> 8,13,13
    h = F.max_pooling(h, (2,2), (2,2))
    # Tanh_2
    h = F.tanh(h)
    # Affine -> 10
    h = PF.affine(h, (10,), name='Affine')
    # Tanh_3
    h = F.tanh(h)
    # Affine_2 -> 1
    h = PF.affine(h, (1,), name='Affine_2')
    # Sigmoid
    h = F.sigmoid(h)
    # BinaryCrossEntropy
    # h = F.binary_cross_entropy(h, y)
    return h


class Node:
	def __init__(self, xy, r):
		self.xy = xy
		self.r = r
		self.value = None
	
	def isReady(self):
		return self.value >= 0

	def setType(self, value): #Options A/B/AB
		self.value = value

def squareCenterInSquare(square1, square2):
	center1 = (square1[0] + (square1[2] / 2), square1[1] + (square1[2] / 2))
	center2 = (square2[0] + (square2[2] / 2), square2[1] + (square2[2] / 2))
	return math.hypot((center1[0] - center2[0]), (center1[1] - center2[1])) < (square2[2]/2)

# same network as classifySymbols(). However, this is used to
def tieBreak(img):
	# load parameters
	nn.clear_parameters()
	nn.load_parameters(MLPATH + "letterOrNumber.h5")

	# Prepare input variable
	lnVar=nn.Variable((1,1,64,64))
	lnVar.data.zero()

	# Build network for inference
	lnAnswer = letterOrNum(lnVar, "", test=True)

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	h,w = img.shape
	
	current = np.copy(img[int(round(0.1 * h)):int(round(0.9 * h)),int(round(0.1 * w)):int(round(0.9 * w))])
	current = cv2.resize(current, (64, 64), interpolation=cv2.INTER_CUBIC)
	current = cv2.medianBlur(current,3)
	ret,current = cv2.threshold(current, 240, 255, cv2.THRESH_BINARY)
	current = current / 255
	lnVar.d = (current)
	lnAnswer.forward()
	ret = lnAnswer.d[0][0]

	nn.clear_parameters()
	return ret

def joinSymbols(img, symbols):
	done = []
	for i in range(len(symbols)):
		done.append(False)
	filtered = []
	rep = 0
	while True:
		target = None
		firstI = -1
		for i in range(len(symbols)):
			if not done[i]:
				firstI = i
				target = symbols[i]
				done[i] = True
				break

		if target != None:
			xList = [target[0]]
			yList = [target[1]]
			sideList = [target[2]]
			contentList = [target[3]]
			for j in range(firstI + 1, len(symbols)):
				current = symbols[j]
				if (not done[j]) and squareCenterInSquare(target, current):
					done[j] = True
					xList.append(current[0])
					yList.append(current[1])
					sideList.append(current[2])
					contentList.append(current[3])
			xMed = int(round(statistics.median(xList)))
			yMed = int(round(statistics.median(yList)))
			sideMed = int(round(statistics.median(sideList)))
			try:
				contentMod = statistics.mode(contentList)
			except:
				if tieBreak(img[yMed:yMed + sideMed, xMed:xMed + sideMed]) >= 0.2:
					contentMod = "Num"
				else:
					contentMod = "Let"
			filtered.append((xMed, yMed, sideMed, contentMod))
		else:
			break
	return(filtered)

def classifyNumber():
	pass

def classifyLetter():
	pass

def classifySymbol(img, symbols):
	# load parameters
	nn.clear_parameters()
	nn.load_parameters(MLPATH + "letterOrNumber.h5")

	# Prepare input variable
	lnVar=nn.Variable((1,1,64,64))
	lnVar.data.zero()

	# Build network for inference
	lnAnswer = letterOrNum(lnVar, "", test=True)

	img = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
	labeled = []
	for symbol in symbols:
		current = np.copy(img[symbol[1]:symbol[1] + symbol[2], symbol[0]:symbol[0] + symbol[2]])
		# print("side: {}, x: {} y: {}".format(side, x, y))
		current = cv2.resize(current, (64, 64), interpolation=cv2.INTER_CUBIC)
		current = cv2.medianBlur(current,3)
		ret,current = cv2.threshold(current, 240, 255, cv2.THRESH_BINARY)
		current = current / 255

		lnVar.d = (current)
		lnAnswer.forward()
		if lnAnswer.d[0][0] < 0.025:
			labeled.append((symbol[0], symbol[1], symbol[2], "Let"))
		elif lnAnswer.d[0][0] >= 0.975:
			labeled.append((symbol[0], symbol[1], symbol[2], "Num"))

	nn.clear_parameters()
	return labeled

STEP = 0.05
SCALE = 0.8
CIRCLEAMP = 1.5
def recognize(img):
	# load parameters
	nn.clear_parameters()
	nn.load_parameters(MLPATH + "binaryFind.h5")

	# Prepare input variable
	binaryVar=nn.Variable((1,1,64,64))
	binaryVar.data.zero()

	# Build network for inference
	binaryAnswer = binaryTargeting(binaryVar, "", test=True)

	img = np.copy(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h, w = img.shape
	detectList = [] 
	side = int(round(0.2 * min(h, w)))
	while side >= (0.08 * min(h, w)):
		microstep = int(round(side * STEP))
		for x in range(0, w - side - 1, microstep):
			for y in range(0, h - side - 1, microstep):
				current = np.copy(img[y:y + side, x:x + side])
				# print("side: {}, x: {} y: {}".format(side, x, y))
				current = cv2.resize(current, (64, 64), interpolation=cv2.INTER_CUBIC)
				current = cv2.medianBlur(current,3)
				ret,current = cv2.threshold(current, 240, 255, cv2.THRESH_BINARY)
				current = current / 255

				binaryVar.d = (current)
				binaryAnswer.forward()
				if binaryAnswer.d[0][0] < 0.0002:
					smaller = int(round(0.75*side))
					current = np.copy(img[y:y + smaller, x:x + smaller])
					# print("side: {}, x: {} y: {}".format(side, x, y))
					current = cv2.resize(current, (64, 64), interpolation=cv2.INTER_CUBIC)
					current = cv2.medianBlur(current,3)
					ret,current = cv2.threshold(current, 240, 255, cv2.THRESH_BINARY)
					current = current / 255

					binaryVar.d = (current)
					binaryAnswer.forward()
					if binaryAnswer.d[0][0] < 0.0002:
						# print("Confidence: {}".format(binaryAnswer.d[0][0]))
						# print("TESS: {}".format(ImageUtils.tesseractOCR(img[y:y + side, x:x + side])))
						# ImageUtils.show(img[y:y + side, x:x + side])
						# img[y:y + side, x:x + side] = 255
						currentNode = (x, y, int(round(0.8 * side)))
						detectList.append(currentNode)
		side = int(round(SCALE * side))
	
	nn.clear_parameters()
	return detectList

def main():
	colorList = {'Let': (255,0,0), 'Num': (0,0,255), 'Let-Num': (0,255,0)}
	img = cv2.imread('../Examples/BigTest.jpg')
	print("Detecting Text Region...\n")
	symbols = recognize(img)
	print("Classifying Text Region as Letter or Text...\n")
	# symbols = classifySymbol(img, symbols)
	print("Joining Duplicate Symbols...\n")
	# symbols = joinSymbols(img, symbols)
	if symbols is not None:
		print('{} Circles Found.'.format(len(symbols)))
		for i in symbols:
			cv2.rectangle(img, (i[0], i[1]), (i[0] + i[2], i[1] + i[2]), colorList['Let-Num'], 2)

	else:
		print('No Circles Found!')
		 
	ImageUtils.show(img)


if __name__ == '__main__':
	main()