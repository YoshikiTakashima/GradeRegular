import cv2
import numpy as np
import statistics
from matplotlib import pyplot as plt

import ImageUtils

class Circle:
    def __init__(self, xy, r):
        self.xy = xy
        self.r = r
        self.value = -1 #negative value indecates unset
    
    def isReady(self):
        return self.value >= 0

    def setIsState(self, value): #value is 0 if 0-loop, 1 if 1-loop, -1 if start state, 3 if final state, 2 else.
        self.value = value

def filterRepetition(circles):
    filtered = []
    donelist = []
    while len(donelist) < len(circles):
        remaining = list(set(range(len(circles))) - set(donelist))
        i = remaining[0]
        xList = []
        yList = []
        rList = []
        for rIndex in remaining:
            if int(ImageUtils.distanceBetween((circles[i].xy[0], circles[i].xy[1]), 
            (circles[rIndex].xy[0], circles[rIndex].xy[1]))) <= (circles[i].r):
                xList.append(circles[rIndex].xy[0])
                yList.append(circles[rIndex].xy[1])
                rList.append(circles[rIndex].r)
                donelist.append(rIndex)
        xMed = int(round(statistics.median(xList)))
        yMed = int(round(statistics.median(yList)))
        rMed = int(round(statistics.median(rList)))
        filtered.append(Circle((xMed, yMed), rMed))
        #print("FILTERING {}".format(remaining))
    return(filtered)

def classifyContents(image, circles):
    h, w = image.shape
    i = 0
    for c in circles:
        y = c.xy[1]
        x = c.xy[0]
        shift= int(0.5*c.r)
        cropped = image[y - shift:y + shift, x - shift:x + shift]
        ImageUtils.show(cropped)
        ImageUtils.ocrToString(cropped)
    return(circles)

def recognize(img):
    img = ImageUtils.denoiseMedian(img)
    img = ImageUtils.toGreyImg(img)

    height, width = img.shape
    # print("HxW = {}x{}".format(height, width)) 

    minRad = round(0.1 * min(height, width))
    maxRad = round(0.25 * max(height, width))

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                param1=50,param2=30,minRadius=minRad,maxRadius=maxRad)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        myCircles = []
        for i in circles[0,:]:
            myCircles.append(Circle((i[0],i[1]), i[2]))
        circles = filterRepetition(myCircles)
        circles = filterRepetition(circles)
        circles = classifyContents(img, circles)
    else:
        circles = None
    return circles

def main():
    img = cv2.imread('./Examples/handDFA.jpg')
    circles = recognize(img)

    if circles is not None:
        print('{} Circles Found.'.format(len(circles)))
        for i in circles:
            # draw the outer circle
            cv2.circle(img,(i.xy[0],i.xy[1]),i.r,(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i.xy[0],i.xy[1]),2,(0,0,255),3)
    else:
        print('No Circles Found!')
         
    ImageUtils.show(img)


if __name__ == '__main__':
    main()