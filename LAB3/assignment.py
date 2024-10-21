import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def search(a,arr):
    for i in range(1,255):
        if(a == arr[i]):
            return i
        elif (a < arr[i]):
            b = arr[i]
            c = arr[i-1]
            if((b-a)>(a-c)):
                return i-1
            else:
                return i
    return 255

def gaussian(m,sigma):
    constant = 2*sigma*sigma
    cons = 1/(np.sqrt(2*3.1416)*sigma)
    g = np.empty(shape = 256)
    for i in range(256):
        g[i] = np.exp(-((i-m)**2)/(constant))*cons
    return g
    
u1,sigma1 = [int(x) for x in input('Enter the values of miu1 and sigma1:').split()]
u2,sigma2 = [int(x) for x in input('Enter the values of miu2 and sigma2:').split()]

g1 = gaussian(u1, sigma1)
g2 = gaussian(u2, sigma2)

g3 = g1+g2


plt.title(label="Bimodal Function",
          fontsize=20,
          color="black")
plt.plot(g3)
plt.show()

#print(g1)
#print(g2)
#print(g3)

totalg = np.sum(g3)
pdfg = (g3)/totalg

cdfg = np.zeros(256)
cdfg[0] = pdfg[0]

for i in range(1,len(pdfg)):
    cdfg[i] = cdfg[i-1] + pdfg[i]
    
cdfg = cdfg*255

plt.title(label="CDF of Bimodal Function",
          fontsize=20,
          color="black")
plt.plot(cdfg)
plt.show()

cdfg = np.round(cdfg).astype(np.uint8)

img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original",img)
plt.imshow(img,cmap='gray')
plt.show()
plt.title(label="Histogram of Input Image",
          fontsize=20,
          color="black")
plt.hist(img.ravel(),256,[0,256])
plt.show()

h,w = img.shape
#print(h,w)

total = h*w
freq = np.zeros(256)

for i in range(h):
    for j in range(w):
        c = img[i][j]
        freq[c] = freq[c]+1

#print(freq)

pdf = freq/total
#print(pdf)

cdf = np.zeros(256,dtype = float)

for i in range(255):
    cdf[i+1] =cdf[i] + pdf[i]

cdf = np.around(cdf*255)

#print(cdf)

for j in range(255):
    cdf[j]=cdf[j+1]

#print(cdf)

output = np.zeros_like(img)

for i in range(h):
    for j in range(w):
        t = img[i][j]
        q = cdf[t]
        output[i][j] = q

cv2.imshow("Equalized",output)


matching = np.zeros_like(img)

for i in range(h):
    for j in range(w):
        a = cdf[img[i,j]]
        b = search(a,cdfg)
        matching[i,j] = b
cv2.imshow("Matching",matching)

intensity = np.zeros(256,dtype = float)

r,t = matching.shape

for i in range(r):
    for j in range(t):
        p = matching[i][j]
        intensity[p] = intensity[p]+1

frame = r*t

pd = intensity/frame
cd = np.zeros(256,dtype = float)

cd[0] = 0

for i in range(255):
    cd[i+1] = cd[i] + pd[i]

cd = np.around(255*cd)

plt.title(label="CDF of Output Image",
          fontsize=20,
          color="red")

plt.plot(cd)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()




