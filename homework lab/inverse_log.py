import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg",0)
img = cv2.resize(img,(700,700))
cv2.imshow("original",img)

c = 255/(np.log(1+255))
inv_log_img = np.exp(img**1/c)-1
inv_log_img = np.array(inv_log_img, dtype = np.uint8)

cv2.imshow("INVERSE LOG",inv_log_img)
cv2.waitKey()
cv2.destryAllWindows()
