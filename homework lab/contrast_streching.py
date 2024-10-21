


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpg")
img = cv2.resize(img,(700,700))
cv2.imshow("original",img)

# Load image

# Split image into color channels
b, g, r = cv2.split(img)
# Stretch each color channel separately
b_stretched = cv2.equalizeHist(b)
g_stretched = cv2.equalizeHist(g)
r_stretched = cv2.equalizeHist(r)

# Merge the stretched color channels back into an RGB image
stretched = cv2.merge((b_stretched, g_stretched, r_stretched))

cv2.imshow("KJDH",stretched)


cv2.waitKey(0)
cv2.destroyAllWindows()