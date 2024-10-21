import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("b.jpeg")

b,g,r = cv.split(img)

q = img[:,:,0]
t = img[:,:,1]
n = img[:,:,2]

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
plt.subplot(2, 2, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'b')

img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])
plt.subplot(2, 2, 4)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
plt.plot(histr,color = 'g')



def equalization(img):
    freq = cv.calcHist([img],[0],None,[256],[0,255])
    plt.plot(freq)
    pdf = np.zeros(256,np.float64)
    total = img.shape[0]*img.shape[1]
    
    for i in range(0,256):
        pdf[i] = freq[i]/total
    
    cdf = np.zeros(257,np.float64)
    
    cdf[0] = 0
    
    for i in range (1,257):
        cdf[i]+= cdf[i-1]+pdf[i-1]
    
    s = np.zeros(256,np.float64)
    
    for i in range(0,256):
        s[i] = np.around((cdf[i+1])*255)
    
    output = np.zeros((img.shape),np.float64)
    
    for i in range (0, img.shape[0]):
        for j in range (0,img.shape[1]):
            output[i][j] = s[img[i][j]]
            
    return output
    

lal = equalization(b)
shobuj = equalization(g)
nil = equalization(r)

merged = cv.merge((lal,shobuj,nil))
cv.imshow("Merged",merged)






cv.waitKey(0)
cv.destroyAllWindows()
