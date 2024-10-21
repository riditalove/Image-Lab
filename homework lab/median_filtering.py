import cv2
import numpy as np
import matplotlib.pyplot as plt

img_noisy1 = cv2.imread("lena.jpg",0)
img_noisy1 = cv2.resize(img_noisy1,(700,700))
cv2.imshow("original",img_noisy1)

m, n = img_noisy1.shape

img_new1 = np.zeros([m, n])

for i in range(1, m-1):
	for j in range(1, n-1):
		temp = [img_noisy1[i-1, j-1],
			img_noisy1[i-1, j],
			img_noisy1[i-1, j + 1],
			img_noisy1[i, j-1],
			img_noisy1[i, j],
			img_noisy1[i, j + 1],
			img_noisy1[i + 1, j-1],
			img_noisy1[i + 1, j],
			img_noisy1[i + 1, j + 1]]
		
		temp = sorted(temp)
		img_new1[i, j]= temp[4]

img_new1 = img_new1.astype(np.uint8)
cv2.imshow('new_median_filtered.png', img_new1)

cv2.waitKey()
cv2.destroyAllWindows()