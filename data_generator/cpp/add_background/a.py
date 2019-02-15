import cv2
import numpy as np
img = cv2.imread('/home/zj/temp/0/0_1137.jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow('1',gray)
cv2.waitKey()