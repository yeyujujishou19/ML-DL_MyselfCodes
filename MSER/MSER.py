import cv2
import matplotlib.pyplot as plt
import numpy as np
import nms

img = cv2.imread('D:/11.png')
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create(_min_area=30, _max_area=600)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)
keep = []
for box in boxes:
    x, y, w, h = box
    keep.append([x, y, x + w, y + h])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# plt.imshow(img, 'brg')
# plt.show()

keep2=np.array(keep)
pick = nms.nms(keep2, 0.3)
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 0, 0), 1)

cv2.imwrite("D:/33.png", orig)
cv2.imshow("After NMS", orig)
cv2.waitKey(0)