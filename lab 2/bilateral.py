

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.jpg', 0)


def bilateral(kernel):
    out = np.zeros(img.shape, dtype=np.float32)
    #padded image
    replicate = cv2.copyMakeBorder(img, center_x,center_x,center_y,center_y, cv2.BORDER_REPLICATE)
    replicate=replicate/255
    for k in range (center_x, replicate.shape[0]-center_x):
        for l in range(center_y, replicate.shape[1]-center_y):
            kernel2 = np.zeros(kernel.shape, dtype=np.float32)
            for i in range (-center_x, center_x+1):
                for j in range (-center_y, center_y+1):
                    power = (replicate[i][j] - replicate[k+i][l+j])
                    power = power*power/(2*sigma*sigma)
                    val = np.exp(-power)
                    kernel2[center_x-i][center_y-j] = val*c * kernel[center_x-i][center_y-j]
            for p in range (-center_x, center_x+1):
                for q in range (-center_y, center_y+1):
                    out[k-center_x][l-center_y] += replicate[k+i][l+j] * kernel2[center_x-i][center_y-j]
    plt.imshow(img, 'gray')
    plt.title('Input')
    plt.show()
    out=out*255
    plt.imshow(out, 'gray')
    plt.title('Bilateral Output')
    plt.show()


sigma = int(input('Value of sigma: '))
ker_width = 5*sigma
ker_height = 5*sigma
kernel = np.zeros((ker_width, ker_height), dtype = np.float32)

center_x = ker_width//2
center_y = ker_height//2
c = 2*3.1416*sigma*sigma
c = 1/c
for i in range (-center_x, center_x+1):
    for j in range(-center_y, center_y+1):
        power = (i*i + j*j)/(2*sigma*sigma)
        val = np.exp(-power)
        kernel[center_x+i][center_y+j] = val*c
kernel /= kernel.sum()
#print(kernel)

bilateral(kernel)



