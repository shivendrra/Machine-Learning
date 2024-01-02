import os
os.chdir('D:/Machine Learning/Machine Learning Git-repo/Computer Vision')
import cv2 as cv

img = cv.imread('Sample photos/img-1.jpg')

def rescale_image(frame, scale=0.2):
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)

  return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

rescaled_image = rescale_image(img)
cv.imshow('rescaled img', rescaled_image)
cv.waitKey(0)