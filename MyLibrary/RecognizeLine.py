import cv2
import numpy as np
from random import randint

import ImageUtils



class Line:
	def __init__(self, cvLine):
		self.cvLine = cvLine
		self.value = -1
	
	def isReady(self):
		return (self.value >= 0) and (self.value < 2)

def recognize(img, circles):
	
	return 1


def main():
	img = cv2.imread('./Examples/handDFA.jpg')
	lines = recognize(img, 0)

	if lines is not None:
		print('{} Lines Found.'.format(len(lines)))
	else:
		print('No Lines Found!')
	
	img = cv2.resize(img, (1320, 1020))    
	ImageUtils.show(None)

if __name__ == '__main__':
	main()