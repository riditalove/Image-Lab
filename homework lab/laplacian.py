import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.jpg",0)
cv2.imshow("original",img)


kernal = np.array(([0,-1,0],
              [-1,4,-1],
              [0,-1,0]),np.float32)

def convolve(image,kernal):
    k_h,k_w = kernal.shape
    im_h,im_w = image.shape
    h=k_h//2
    w=k_w//2
    image = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_REPLICATE)
    print(image.shape)
    res_img = np.array(image)
    for i in range(h,im_h-h):
        for j in range(w,im_w-w):
            sum = 0
            for m in range(k_h):
                for n in range(k_w):
                    sum = sum + kernal[m][n]*image[i-h+m][j-w+n]
            res_img[i][j] = sum
            
    return res_img


image = convolve(img,kernal)
image=np.round(image).astype(np.uint8)
cv2.normalize(image,image, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("okay",image)
cv2.waitKey(0)
cv2.destroyAllWindows()