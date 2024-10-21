import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lena.jpg",0)
img = cv2.resize(img,(700,700))
cv2.imshow("original",img)

def mean(img):
   
  
    img = cv2.copyMakeBorder(src=img, top=1, bottom=1, left=1, right=1,borderType= cv2.BORDER_CONSTANT)#BORDER_WRAP, cv.BORDER_REFLECT  
    m = img.shape[0]
    n = img.shape[1]
    print(m)
    print(n)
    mean_img =  np.zeros(img.shape,np.uint8)    
    result = 0
    for i in range(1,m-1):
        for j in range(1,n-1):
            for y in range(-1,2):
                for x in range(-1,2):
                    result = result + img[i+y,j+x]
            mean_img[i][j] = int(result/9)
            result = 0
                    
    return mean_img

mean_img = mean(img)
cv2.imshow("mean",mean_img)
cv2.waitKey()
cv2.destroyAllWindows()
                   
                    
            
    
    
    
    




cv2.waitKey(0)
cv2.destroyAllWindows()

