import cv2
import numpy as np
import math
import statistics
from matplotlib import pyplot as plt

import MyLibrary.ImageUtils as ImageUtils
import MyLibrary.VerilogUtil as VUtil

SCANNERWIDTH = 4 # NOTE: Real scanner width is 2(that value) + 1
def scanToDark(img, point, minSize, maxSize):
	x = point[0]
	y = point[1]
	h, w = img.shape

	top = 1
	topHitSide = False
	bot = 1
	botHitSide = False
	left = 1
	leftHitSide = False
	right = 1
	rightHitSide = False

	while (y - top) >= 0:
		allDark = True
		for i in range(-1 *SCANNERWIDTH, SCANNERWIDTH + 1):
			allDark = allDark and (img[(y - top)][(x + i)] < 128)

		if allDark:
			break
		if (y - top) == 0:
			topHitSide = True
		top += 1

	while (y + bot) < h:
		allDark = True
		for i in range(-1 *SCANNERWIDTH, SCANNERWIDTH + 1):
			allDark = allDark and (img[(y + bot)][(x + i)] < 128)

		if allDark:
			break
		if (y + bot) == h - 1:
			botHitSide = True
		bot += 1

	while (x + right) < w:
		allDark = True
		for i in range(-1 *SCANNERWIDTH, SCANNERWIDTH + 1):
			# print("x + i {}//// x + r")
			allDark = allDark and (img[(y + i)][(x + right)] < 128)

		if allDark:
			break
		if (x + right) == w - 1:
			rightHitSide = True
		right += 1

	while (x - left) >= 0:
		allDark = True
		for i in range(-1 *SCANNERWIDTH, SCANNERWIDTH + 1):
			allDark = allDark and (img[(y + i)][(x - left)] < 128)

		if allDark:
			break
		if (x - left) == 0:
			leftHitSide = True
		left += 1

	isContained = not (topHitSide or botHitSide or leftHitSide or rightHitSide)

	vertSize = top + bot
	horizSize = left + right
	isGoodSize = (min(vertSize, horizSize) > minSize) and (max(vertSize, horizSize) < maxSize)

	symmetricTolerance = 0.5 #0.5 orig
	isSymmetric = (abs((vertSize / horizSize) - 1) < symmetricTolerance)

	centerTolerance = 0.175	#0.5 orig
	isCentered = (abs((top - bot) / vertSize) < centerTolerance) and (abs((left - right) / horizSize) < centerTolerance)

	isCircleCenter = isContained and isGoodSize and isSymmetric and isCentered
	return isCircleCenter, int(round(statistics.median([top, bot, left, right])))

def filterRepetition(states):
	done = []
	for i in range(len(states)):
		done.append(False)
	filtered = []

	while True:
		target = None
		firstI = -1
		for i in range(len(states)):
			if not done[i]:
				firstI = i
				target = states[i]
				done[i] = True
				break

		if target != None:
			xList = [target[0]]
			yList = [target[1]]
			rList = [target[2]]
			for j in range(firstI + 1, len(states)):
				current = states[j]
				if (not done[j]) and \
				((math.hypot((current[0] - target[0]), (current[1] - target[1])) < target[2]) or \
				(math.hypot((current[0] - target[0]), (current[1] - target[1])) < current[2])):
					done[j] = True
					xList.append(current[0])
					yList.append(current[1])
					rList.append(current[2])
			xMed = int(round(statistics.median(xList)))
			yMed = int(round(statistics.median(yList)))
			rMed = int(round(statistics.median(rList)))
			filtered.append((xMed, yMed, rMed))
		else:
			break
	return filtered

def orderStates(states):
	result = []
	selectedIndex = 0
	for i in range(1,len(states)):
		selected = states[selectedIndex]
		current = states[i]
		if current[0] < selected[0]:
			selectedIndex = i
	if len(states) > 0:
		leftMost = states[selectedIndex]
		del states[selectedIndex]
		result = [leftMost] + states
	return (result)

def classifyStateType(img, states):
	h,w,depth = img.shape
	classified = []
	for s in states:
		masked = np.copy(img)
		cv2.circle(masked,(s[0],s[1]),int(round(1.13*s[2])),(255, 255, 255), -1)
		masked = ImageUtils.binarize(masked)
		# ImageUtils.show(masked)
		isCircleCenter, r = scanToDark(masked, (s[0], s[1]), 
			int(round(0.1 * min(h, w))), int(round(0.4 * min(h, w))))

		if isCircleCenter:
			classified.append((s[0], s[1], int(round(1.1*r)), 'A'))
		else:
			classified.append((s[0], s[1], s[2], 'NA'))		

	return classified

STEPSCALE = 0.05
def detectStates(img):
	img = np.copy(img)
	img = ImageUtils.binarize(img)
	h, w = img.shape

	stateList = []

	step = int(round(STEPSCALE * min(h, w)))
	TOTAL = int(round(math.ceil(w / step) * math.ceil(h / step)))
	i = 0
	for x in range(0, w - 3, step):
		for y in range(0, h - 3, step):
			isCircleCenter, r = scanToDark(img, (x, y), 
				int(round(0.1 * min(h, w))), int(round(0.4 * min(h, w))))
			if isCircleCenter:
				stateList.append((x, y, r))
			i += 1
			print("Scanning Complete: {:5.5}%".format(100 *i / TOTAL))

	return stateList

def scanLinesAroundStates(img, states):
	SQUARESIZE = 0.025
	STEP = 1
	AMP = 1.5
	SKEW = 1
	img = np.copy(img)
	h, w, depth = img.shape
	lines = []
		
	for j in range(len(states)):
		current = states[j]
		length = int(round(AMP * 2 * current[2]))
		smallLength = int(round(SKEW * length))
		side = int(round(SQUARESIZE * length))
		smallSide = int(round(SQUARESIZE * smallLength))
		topX = max(current[0] - int(round(smallLength / 2)), 0)
		topY = max(current[1] - int(round(length / 2)), 0)
		maxX = min(topX + smallLength - smallSide - 1, w - 1)
		maxY = min(topY + length - side -1, h - 1)

		xList = []
		yList = []

		for y in range(topY, maxY - side, int(round(side * STEP))):
			xList.append(topX)
			yList.append(y)

		for x in range(topX, maxX - smallSide, int(round(smallSide * STEP))):
			xList.append(x)
			yList.append(maxY)

		for y in reversed(range(topY + side, maxY, int(round(side * STEP)))):
			xList.append(maxX)
			yList.append(y)
		
		for x in reversed(range(topX + smallSide, maxX, int(round(smallSide * STEP)))):
			xList.append(x)
			yList.append(topY)

		lastAccepted = False
		for i in range(len(xList)):
			if ImageUtils.avgGreyVal(img[yList[i]: yList[i] + side, xList[i]: xList[i] + smallSide]) < 250:
				if lastAccepted:
					lastRect = lines[- 1][0]
					lastRect = (int(round(statistics.mean([lastRect[0], xList[i]]))), \
						int(round(statistics.mean([lastRect[1], yList[i]]))), \
						int(round(statistics.mean([lastRect[0], xList[i]]))) + smallSide, \
						int(round(statistics.mean([lastRect[1], yList[i]]))) + side)
				else:
					lines.append([(xList[i], yList[i], xList[i] + smallSide, yList[i] + side, j)])
					lastAccepted = True
			else:
				lastAccepted = False

	return lines

def nearNode(x, y, nodes):
	PROXTHRESH = 1.25
	nearNode = False
	for n in nodes:
		dist = math.hypot(x - n[0], y - n[1])
		nearNode = nearNode or (dist < (PROXTHRESH * n[2]))
	return nearNode

def isInRange(x, y, h, w, hSide, wSide):
	isHeightOK = (x >= 0) and (x < (w - wSide - 1))
	isWidthOK = (y >= 0) and (y < (h - hSide - 1))
	return isHeightOK and isWidthOK

def extrapolateLines(img, lines, states):
	DARKNESSTHRSH = 255

	img = np.copy(img)
	h, w, d = img.shape
	result = []

	TOTAL = len(lines)
	count = 0
	for line in lines:
		count += 1

		l = line[-1]
		smallSide = l[2] - l[0]
		side = l[3] - l[1]
		s = states[l[4]]

		img[l[1]:l[3], l[0]:l[2]] = (255, 255, 255)

		neighbors = []
		for vDir in range(-1,2):
			for hDir in range(-1,2):
				boxX = l[0] + (hDir * smallSide)
				boxY = l[1] + (vDir * side)
				neighbors.append(((boxX, boxY, boxX + smallSide, boxY + side), \
				math.hypot(boxX - s[0], boxY - s[1]), \
				ImageUtils.avgGreyVal(img[boxY:boxY + side, boxX:boxX + smallSide])))
				line.append((boxX, boxY, boxX + smallSide, boxY + side))

		top = []
		for n in neighbors:
			if len(top) == 0:
				if n[2] < DARKNESSTHRSH:
					top = n
			else:
				if top[1] < n[1] and n[2] < DARKNESSTHRSH:
					top = n
		if len(top) ==0 :
			continue
		else:
			sq = top[0]
			line.append((sq[0], sq[1], sq[2], sq[3]))
			# result.append(line)
		
		while True:
			l = line[-1]
			smallSide = l[2] - l[0]
			side = l[3] - l[1]

			if not isInRange(l[0] + smallSide, l[1] + side, h, w, side, smallSide):
				break
			if not isInRange(l[0] - smallSide, l[1] - side, h, w, side, smallSide):
				break

			img[l[1]:l[3], l[0]:l[2]] = (255, 255, 255)

			neighbors = []
			for vDir in range(-1,2):
				for hDir in range(-1,2):
					boxX = l[0] + (hDir * smallSide)
					boxY = l[1] + (vDir * side)
					neighbors.append(((boxX, boxY, boxX + smallSide, boxY + side), \
					math.hypot(boxX - s[0], boxY - s[1]), \
					ImageUtils.avgGreyVal(img[boxY:boxY + side, boxX:boxX + smallSide])))
					# line.append((boxX, boxY, boxX + smallSide, boxY + side))

			top = []
			for n in neighbors:
				if len(top) == 0:
					if n[2] < DARKNESSTHRSH:
						top = n
				else:
					if top[1] < n[1] and n[2] < DARKNESSTHRSH:
						top = n
			
			if len(top) == 0:
				print("Partial found: \t{:5.5}%".format(100 * count / TOTAL))
				result.append(line)
				break
			else:
				sq = top[0]
				line.append((sq[0], sq[1], sq[2], sq[3]))
				if nearNode(sq[0], sq[1], states):
					result.append(line)
					result.append([(sq[0], sq[1], sq[2], sq[3])])
					print("NORMAL EXIT: \t{:5.5}%".format(100 * count / TOTAL))
					break
	return result

def connectLines(lines):
	TOTAL = len(lines)
	count = 0

	done = []
	for i in range(TOTAL):
		done.append(False)
	connected = []

	joinWith = -1
	for i in range(TOTAL):
		count += 1
		if done[i]:
			continue
		else:
			done[i] = True
		maxDist = abs(lines[i][0][1] - lines[i][0][3])
		for j in range(len(lines)):
			if not done[j]:
				for cRect in reversed(lines[i]):
					for tRect in reversed(lines[j]):
						if math.hypot(cRect[0] - tRect[0], cRect[1] - tRect[1]) < 2*maxDist:
							joinWith = j
							break
					if joinWith >= 0:
						break
				if joinWith >= 0:
					break
		if joinWith >= 0:
			done[joinWith] = True
			connected.append(lines[i] + list(reversed(lines[joinWith])))
			print("Concat Line: \t{:5.5}%".format(100 * count / TOTAL))
			joinWith = -1
		else:
			print("Filtered Line: \t{:5.5}%".format(100 * count / TOTAL))
	return connected

def detectLabels(img, tracedLines, states):
	letterRects = []
	editImg = img.copy()
	for line in tracedLines:
		for rect in line:
			cv2.rectangle(editImg, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), -1)
	for state in states:
		cv2.circle(editImg,(state[0],state[1]),int(round(1.25 * state[2])),(255, 255, 255),-1)
	
	letterRects = ImageUtils.mser(editImg)
	letterRects = ImageUtils.filterInvalidTextRegions(img, letterRects)

	labeledRects = []
	for rect in letterRects:
		labeledRects.append((rect[0], rect[1], rect[2], rect[3], \
		ImageUtils.ocr01E(img[rect[1]:rect[3], rect[0]:rect[2]])))
	return labeledRects

def nearestElementIndex(x, y, elements):
	fX = x
	fY = y
	fStateIndex = 0
	for i in range(1, len(elements)):
		current = elements[i]
		cX = current[0]
		cY = current[1]
		selected = elements[fStateIndex]
		sX = selected[0]
		sY = selected[1]
		if math.hypot((fX - cX), (fY - cY)) < math.hypot((fX - sX), (fY - sY)):
			fStateIndex = i
	return fStateIndex

def encode(states, lines, labels, img): #img included for debug, remvoe later
	encodedNodeList = []
	
	fList = []
	lList = []
	s2sTransitions = []
	for line in lines:
		first = line[0]
		fList.append(first)
		fX = first[0]
		fY = first[0]
		fStateIndex = nearestElementIndex(fX, fY, states)
		last = line[-1]
		lList.append(last)
		lX = last[0]
		lY = last[1]
		lStateIndex = nearestElementIndex(lX, lY, states)
		s2sTransitions.append((fStateIndex, lStateIndex))

	labelDone = []
	for line in labels:
		labelDone.append(False)
	transitionDone = []
	for t in s2sTransitions:
		transitionDone.append(False)
	
	labeledTransitions = []
	while True: #process all self loops first
		targetIndex = -1
		for j in range(len(s2sTransitions)):
			current = s2sTransitions[j]
			if current[0] == current[1]:
				targetIndex = j
				break
		if targetIndex >= 0:
			target = lines[targetIndex]
			tFirst = target[0]
			tLast = target[-1]
			tX = int(round((tFirst[0] + tLast[0]) / 2))
			tY = int(round((tFirst[1] + tLast[1]) / 2))
			tLabelIndex = nearestElementIndex(tX, tY, labels)

			tValue = labels[tLabelIndex][4]
			labelDone[tLabelIndex] = True
			transit = s2sTransitions[targetIndex]
			transitionDone[targetIndex] = True
			s2sTransitions[targetIndex] = ((transit[0], -1), (transit[1], tValue))
		else:
			break

	for i in range(len(labels)):
		label = labels[i]
		if not labelDone[i]:
			initIndex = -1
			for k in range(len(transitionDone)):
				if not transitionDone[k]:
					initIndex = k
					break
			
			currentClosestIndex = initIndex
			if initIndex >= 0:
				lastLine = lines[currentClosestIndex]
				labelX = label[0]
				labelY = label[1]
				fOrLIndex = nearestElementIndex(labelX, labelY, [lastLine[0], lastLine[-1]])
				for j in range(initIndex + 1, len(lines)):
					if not transitionDone[j]:
						current = lines[j]
						cFirst = current[0]
						cLast = current[-1]
						lastLine = lines[currentClosestIndex]
						
						nearest = nearestElementIndex(labelX, labelY, [lastLine[0], lastLine[-1], cFirst, cLast])
						if nearest >= 2:
							currentClosestIndex = j
							fOrLIndex = nearest - 2
				
				transitionDone[currentClosestIndex] = True
				transit = s2sTransitions[currentClosestIndex]
				labelDone[i] = True
				tValue = label[4]

				encoded = ()
				if fOrLIndex == 0:
					encoded = ((transit[0], tValue), (transit[1], -1))
				else:
					encoded = ((transit[0], -1), (transit[1], tValue))
				s2sTransitions[currentClosestIndex] = encoded
				
				# pImg = img.copy()
				# line = lines[currentClosestIndex]
				# for i in range(len(line)):
				# 	rect = line[i]
				# 	cv2.rectangle(pImg, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)
				# r = label
				# cv2.rectangle(pImg, (r[0], r[1]), (r[2], r[3]), (200,200,0), 2)
				# ImageUtils.show(pImg)

			else:
				break

	return s2sTransitions

def transitionSetToStateValueMap(state, transitionsProcessed):
	stateValueMap = []
	for i in range(len(state)):
		dests = ([], [], [])
		for t in transitionsProcessed:
			if (i, -1) in t:
				index = t.index((i, -1))
				target = ()
				if index == 0:
					target = t[1]
				else:
					target = t[0]
				dests[target[1]].append(target[0])
		stateValueMap.append(dests)
	
	return stateValueMap

def processImage(path):
	img = cv2.imread(path)

	print("Detecting States... ")
	states = detectStates(img)
	states = filterRepetition(states)
	states = filterRepetition(states)
	states = orderStates(states)
	states = classifyStateType(img, states)
	print("\nScanning Lines between states...")
	lines = scanLinesAroundStates(img, states)
	extLines = extrapolateLines(img, lines, states)
	print("Connecting Partial lines to form full lines...")
	lineResult = connectLines(extLines)

	print("\nDetecting Transition Labels...")
	labels = detectLabels(img, lineResult, states)

	print("\nEncoding All Data to Transition Map...")
	encoding = encode(states, lineResult, labels, img)
	finalEncoding = transitionSetToStateValueMap(states, encoding)

	return states, finalEncoding

def main():
	COLORMAP = {'NA': (0, 0, 255), 'A': (255, 0, 0)}
	from sys import argv
	img = cv2.imread(argv[1])

	print("Detecting States... ")
	states = detectStates(img)
	states = filterRepetition(states)
	states = filterRepetition(states)
	states = orderStates(states)
	states = classifyStateType(img, states)
	print("\nScanning Lines between states...")
	lines = scanLinesAroundStates(img, states)
	extLines = extrapolateLines(img, lines, states)
	print("Connecting Partial lines to form full lines...")
	lineResult = connectLines(extLines)

	print("\nDetecting Transition Labels...")
	labels = detectLabels(img, lineResult, states)

	print("\nEncoding All Data to Transition Map...")
	encoding = encode(states, lineResult, labels, img)
	finalEncoding = transitionSetToStateValueMap(states, encoding)

	for e in finalEncoding:
		print(e)
	# print(lines)
	for s in states:
		cv2.circle(img,(s[0],s[1]),int(round(s[2])),COLORMAP[s[3]],2) # draw the outer circle
		cv2.circle(img,(s[0],s[1]),2,(0,0,255),3) # draw the center of the circle
		# ImageUtils.show(img)

	for j in range(len(lineResult)):
		line = lineResult[j]
		for i in range(len(line)):
			rect = line[i]
			cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)
		# print(encoding[j])
		# ImageUtils.show(img)
	
	ANSLIST = ['0', '1', 'E']
	for r in labels:
		# print("LABEL: {}".format(ANSLIST[r[4]]))
		cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (200,200,0), 2)
	ImageUtils.show(img)

	print(VUtil.transitionToVerilog(states, finalEncoding))

	

if __name__ == '__main__':
	main()