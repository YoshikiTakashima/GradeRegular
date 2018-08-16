import cv2
import numpy as np
from random import randint

import ImageUtils

minLineLength = 100
maxLineGap = 10

lsd = cv2.createLineSegmentDetector(1)

class Line:
    def __init__(self, cvLine):
        self.cvLine = cvLine
        self.value = -1
    
    def isReady(self):
        return (self.value >= 0) and (self.value < 2)

def recognize(img):
    img = ImageUtils.toGreyImg(img)
    return lsd.detect(img)[0]


def main():
    img = cv2.imread('./Examples/lines.jpg')
    lines = recognize(img)

    if lines is not None:
        print('{} Lines Found.'.format(len(lines)))
        drawn_img = lsd.drawSegments(img,lines)
    else:
        print('No Lines Found!')
    
    ImageUtils.show(drawn_img)

if __name__ == '__main__':
    main()