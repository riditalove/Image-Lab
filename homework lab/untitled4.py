
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.jpg",0)
cv2.imshow("original",img)


kernal = np.array(([[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]]),np.float32)

def flipped(kernal):
    cpy_kernal = kernal.copy()
    for i in range(kernal.shape[0]):
        for j in range(kernal.shape[1]):
            cpy_kernal[i][j] = kernal[kernal.shape[0]-i-1][kernal.shape[1]-j-1]
    return cpy_kernal

def convolve(image,kernal):
    kernal = flipped(kernal)
    #cpy_kernal = kernal.copy()
    k_h,k_w = kernal.shape
    im_h,im_w = image.shape
    h=k_h//2
    w=k_w//2
    image = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_REPLICATE)
    res_img = np.zeros(image.shape)
    for i in range(h,im_h-h):
        for j in range(w,im_w-w):
            sum = 0
            for m in range(k_h):
                for n in range(k_w):
                    sum = sum + kernal[m][n]*image[i-h+m][j-w+n]
            res_img[i][j] = sum
            
    return res_img


image = convolve(img,kernal)
cv2.imshow("okay",image)
cv2.waitKey(0)
cv2.destroyAllWindows()