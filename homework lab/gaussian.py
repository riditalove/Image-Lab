import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.jpg",0)
cv2.imshow("original",img)


def gaussian_filter(image, sigma):
    ksize = int(4 * sigma + 1)
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    
    for i in range(ksize):
        for j in range(ksize):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            
    kernel = kernel / np.sum(kernel)
    filtered_image = convolution(image, kernel)
    
    return filtered_image

    

def convolution(img,kernel):
   
    im_H = img.shape[0]
    im_W = img.shape[1]
    ker_h = kernel.shape[0]
    ker_w = kernel.shape[1]
    padding = ker_h//2
    h = ker_h//2
    w = ker_w//2
    image_pad = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    image_conv = np.zeros(image_pad.shape,np.float32)
    
    for i in range(h,img.shape[0]-h):
        for j in range(w,img.shape[1]-w):
            sum = 0
            
            for m in range(ker_h):
                for n in range(ker_w):
                    
                    sum = sum + kernel[m][n]*image_pad[i-h+m][j-w+n]
                    
            image_conv[i][j] = sum
            
    return image_conv


sigma = 5

image = gaussian_filter(img, sigma)

cv2.imshow("gaus",image)
cv2.waitKey()
cv2.destroyAllWindows()

            
                

    


    
    


