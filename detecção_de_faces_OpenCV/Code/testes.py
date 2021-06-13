# Testes
import cv2
#import matplotlib
#import numpy

#print(cv2.__version__)

img = cv2.imread('C:\\Users\\jairp\\Desktop\\opencv_python.jpg')
imgCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", img)
cv2.imshow("PB", imgCinza)
cv2.waitKey()
