import os
os.chdir('D:/Machine Learning/Machine Learning Git-repo/Computer Vision')
import cv2 as cv

img_1 = cv.imread('Sample photos/img-1.jpg')
cv.imshow('Main Image', img_1)

def rescale_img(frame, scale=0.50):
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)

  return cv.resize(img_1, dimensions, interpolation=cv.INTER_AREA)

rescaled_img = rescale_img(img_1)
cv.imshow('Rescaled Image', rescaled_img)

cv.waitKey(0)