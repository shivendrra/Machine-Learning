import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

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

# video capture
capture = cv.VideoCapture('Sample videos/video_1.mp4')

while True:
  isTrue, frame = capture.read()
  cv.imshow('video', frame)

  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()