import numpy as np
import cv2

def getLimits(color):

    c = np.uint8([[color]]) #Used to convert bgr values into hsv

    hsvColor  = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimt = hsvColor[0][0][0] - 25, 100, 100
    upperLimt = hsvColor[0][0][0] + 25, 255, 255

    lowerLimt = np.array(lowerLimt, dtype=np.uint8)
    upperLimt = np.array(upperLimt, dtype=np.uint8)

    return lowerLimt, upperLimt
