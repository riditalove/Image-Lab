
import numpy as np
import cv2
import math


def spatial_gaussian_kernel(ksize, sigma,x,y):
    kernal = np.zeros((ksize,ksize),dtype ="float32")
    h = ksize//2
    w = ksize//2
    Xc = h
    Yc = w
    r_square = ((x-0)**2) + ((y-0)**2)
    r = np.sqrt(r_square)
    d_square = ((ksize//2)**2)
    cons = 3/4
    for i in range(-h,h+1):
        for j in range(-w,w+1):
            if(r<=-1 or r>=1):
                kernal[i+h][j+w] = 0            
            else:
                kernal[i+h][j+w] = cons*(1-(r_square/d_square))
    
    return kernal


def range_gaussian_kernel(img, x, y, ksize, sigma):
    kernel = np.zeros((ksize, ksize), dtype="float32")
    gconst = np.sqrt(2 * math.pi) * sigma
    k = ksize // 2
    Ip = img[x][y] 
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            Iq = img[x + i][y + j]
            kernel[i + k][j + k] = (
                math.exp(-((Ip - Iq) ** 2) / (2 * sigma * sigma))
            ) 
    return kernel


def multiplication(sp_kernel, rng_kernel, ksize):
    final_kernel = np.zeros((ksize, ksize), dtype="float32")
    for i in range(ksize):
        for j in range(ksize):
            final_kernel[i][j] = sp_kernel[i][j] * rng_kernel[i][j]

    summ = final_kernel.sum()
    final_kernel = final_kernel / summ
    return final_kernel


img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
# blur=cv2.bilateralFilter(img,5,5,5)
# cv2.imshow("Blur",blur)
# img=img/255
ksize = 5
sigma = 5
height = img.shape[0]
width = img.shape[1]
output = np.zeros((height, width), dtype="float32")
k = ksize // 2
bordered_output = cv2.copyMakeBorder(img, k, k, k, k, cv2.BORDER_REPLICATE)


n = 4
for x in range(height):
    for y in range(width):
        sp_kernel = spatial_gaussian_kernel(ksize,sigma,x,y)
        rng_kernel = range_gaussian_kernel(bordered_output, x + k, y + k, ksize, sigma)
        kernel = multiplication(sp_kernel, rng_kernel, ksize)
        for i in range(ksize):
            for j in range(ksize):
                output[x][y] += (
                    bordered_output[x + i][y + j] * kernel[ksize - 1 - i][ksize - 1 - j]
                )

# output=output*255
cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
output = np.round(output).astype(np.uint8)

cv2.imshow("Actual", img)
cv2.imshow("Biliteral", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
