import os
os.chdir('D:/Machine Learning/Machine Learning Git-repo/Computer Vision')
import cv2 as cv

img_1 = cv.imread('Sample photos/img-1.jpg')
cv.imshow('Sky', img_1)
cv.waitKey(0)