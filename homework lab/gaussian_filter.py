

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

img = cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(cv2.cvtColor(img,0))
plt.show()

im_H = img.shape[0]
im_W = img.shape[1]


def gaussian(sigma,img,ksize,padding):
    gfilter = np.zeros((ksize,ksize),np.float32)
    div = (sigma*sigma)*2
    for i in range(-padding,padding+1):
        for j in range(-padding,padding+1):
            gfilter[i+padding,j+padding] = math.exp(-((i**2+j**2)/div))
    return gfilter



ksize = 7
padding = (ksize-1)//2
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

output_H = (im_H + ksize-1)
output_W = (im_W + ksize-1)

result = np.zeros((output_H,output_W),np.float32)

sigma = 5

div = (sigma*sigma)*2

gaussian_filter = gaussian(sigma,img,ksize,padding)
 
for x in range(padding,output_H-padding):
    for y in range(padding,output_W-padding):
        a = 0
        normalize = 0
        for i in range(-padding,padding+1):
            for j in range(-padding,padding+1):
                a += gaussian_filter[i+padding,j+padding]*img[x-i,y-j]
                normalize += gaussian_filter[i+padding,j+padding]
        result[x,y] = a/normalize
        result[x,y] /= 255


cv2.imwrite('result.png',result)
cv2.imshow("gaussian_filter",result)
cv2.waitKey()
cv2.destroyAllWindows()

