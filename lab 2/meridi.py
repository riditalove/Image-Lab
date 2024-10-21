
import numpy as np
import cv2
import math

img = cv2.imread("cube.png", cv2.IMREAD_GRAYSCALE)

def spatial_distance(ksize,sigma):
    kernal = np.zeros((ksize,ksize),dtype ="float32")
    h = ksize//2
    w = ksize//2
    Xc = h
    Yc = w
    cons = 3/4
    for i in range(ksize):
        for j in range(ksize):
            r_square = ((i-h)**2) + ((j-w)**2)
            r = np.sqrt(r_square)
            d_square = ((ksize//2)**2)
            d = np.sqrt(d_square)
            if(r/d<=-1 or r/d>=1):
                kernal[i,j] = 0                          
            else:
                kernal[i,j] = cons*(1-(r_square/d_square))
                
    return kernal


def pictorial_distance(x,y,ksize,sigma,img):
    kernal = np.zeros((ksize,ksize),dtype = "float32")
    h = ksize//2
    w = ksize//2
    cons = 2*(math.pi)*sigma*sigma
    Ip = img[x,y]
    for i in range(-h,h+1):
        for j in range(-w,w+1):
            c = 2*(sigma**2)
            Iq = img[x+i,y+j]
            dist = abs(Ip - Iq)
            #print(dist)
            kernal[i+h,j+w] = (math.exp(-(dist**2)/c))
    return kernal

def multipication(img,img1,k):
    output = np.zeros((k,k),dtype = "float32")
    for i in range(k):
        for j in range(k):
            output[i,j] = img[i,j]*img1[i,j]
    s = output.sum()
    output = output/s
    return output

def bilateral(img,sigma,k):
    
    bordered_output = cv2.copyMakeBorder(img, k//2, k//2, k//2, k//2, cv2.BORDER_REPLICATE)
    s_dist = np.zeros((k,k),dtype = "float32")
    intesnity = np.zeros((k,k),dtype = "float32")
    x = img.shape[0]
    y = img.shape[1]
    main_kernel = np.zeros((k,k),dtype = "float32")
    out = np.zeros((x,y),dtype = "float32")
    sp_kernel = spatial_distance(k,sigma)
    for i in range(x):
        for j in range(y):  
            intesnity = pictorial_distance(i, j, k, sigma,bordered_output)
            main_kernel = multipication(sp_kernel, intesnity, k)
            for p in range(k):
                for q in range(k):
                    out[i][j] += (
                        bordered_output[p+ i][q + j] * main_kernel[k - 1 - p][k - 1 - q]
                    )
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    
    return out
            
            
            
    
    

output_img = bilateral(img,80,7)
cv2.imshow("Actual", img)
cv2.imshow("loll", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()