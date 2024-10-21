
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img= cv2.imread('b.jpeg')
cv2.imshow("Input",img)
plt.show()
L=256

b,g,r = cv2.split(img)

def his(img):
    histr=cv2.calcHist([img],[0],None,[256],[0,255])
    plt.title("Built In")
    plt.plot(histr)
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Input Image Histogram")
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

    img2=np.zeros_like(img)
    h,w=img.shape
    Tpixel=h*w

    new_hist=histr
    new_hist=new_hist/Tpixel
    cdf = new_hist
    plt.subplot(1,3,2)
    plt.title("CDF Image Histogram")
    plt.hist(cdf.ravel(),256,[0,256])
    plt.show()
    s=new_hist
    sm=0.0
    for i in range(256):
        sm=sm+cdf[i]
        cdf[i]=sm
        s[i]=np.around((L-1)*cdf[i])
    plt.plot(cdf)
    plt.show()
    for i in range(h):
        for j in range(w):
            x=img[i][j]
            img2[i][j]=s[x]
    cv2.imshow("Output",img2)
    plt.show()
    plt.subplot(1,3,3)
    plt.title("Output Image Histogram")
    plt.hist(img2.ravel(),256,[0,256])
    plt.show()
    return img2


b=his(b);
g=his(g);
r=his(r);

merged = cv2.merge((b,g,r))
cv2.imshow("Merged",merged)




img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imshow("img_hsv",img_hsv)

# img_hsv[:,:,0]=his(img_hsv[:,:,0]);
# img_hsv[:,:,1]=his(img_hsv[:,:,1]);

img_hsv[:,:,2]=his(img_hsv[:,:,2]);


image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
cv2.imshow("FINAL",image)

cv2.waitKey(0)

cv2.destroyAllWindows()
