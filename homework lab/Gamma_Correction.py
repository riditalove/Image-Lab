import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg",0)
img = cv2.resize(img,(700,700))
cv2.imshow("original",img)

def adjust_gamma(image,gamma):
    invGamma = 1.0/gamma
    table = np.array([((i/255)**invGamma)*255 for i in np.arange(0,256)])
    lut_img = cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
    return lut_img


gamma = adjust_gamma(img, 1.5)

cv2.imshow("gamma correction",gamma)

cv2.waitKey(0)
cv2.destroyAllWindows()