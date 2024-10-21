

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg",0)
cv2.imshow("original",img)

def contrast_stretching(img):
    h=img.shape[0]
    w=img.shape[1]
    maxiI = np.max(img)
    miniI = np.amin(img)
    contrast = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            r=img[i,j]
            contrast[i,j] = 255*(r-miniI)/(maxiI-miniI)
    contrast=np.round(contrast).astype(np.uint8)
    return contrast


stretched = contrast_stretching(img)
cv2.imshow("contrast streching",stretched)


cv2.waitKey(0)
cv2.destroyAllWindows()
