import os
os.chdir('D:/Machine Learning/Machine Learning Git-repo/Computer Vision')
import cv2 as cv

capture = cv.VideoCapture('Sample videos/video_1.mp4')

while True:
  isTrue, frame = capture.read()
  cv.imshow('video', frame)

  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()