import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img = cv2.imread('b.jpeg')
#cv2.imshow('Input',img)

b,g,r=cv2.split(img)

freq = cv2.calcHist([b],[0],None,[256],[0,255])
freqi = freq.flatten()
h,w = b.shape
total = h*w
pdf = freqi/total
cdf = np.zeros(256,dtype=float)
for i in range(255):
    cdf[i+1] = cdf[i]+pdf[i]

cdf=cdf*255
cdf = np.around(cdf)

for i in range(255):
    cdf[i]=cdf[i+1]

#print(cdf)

output = np.zeros_like(b)
d,p = output.shape
for i in range(d):
    for j in range(p):
        x = b[i][j]
        v = cdf[x]
        output[i][j] = v
        
print(output)

op = cv2.imshow("ok",output)

    

   
    
    
    
#b = equalization(img, 0)  
#g = equalization(img, 1)  
#r = equalization(img, 2)


cv2.waitKey(0)
cv2.destroyAllWindows()
