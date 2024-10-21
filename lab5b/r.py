import cv2
import numpy as np

img = cv2.imread('input.jpg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)
cv2.imshow("Original", img)

kernel = np.array([[0, 0, 0], [1, 1, 0],[1,0,0]])
print(kernel)

kernel1 = (kernel) *255
kernel2 = np.uint8(kernel1)

rate = 50
kernel3 = cv2.resize(kernel2, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
#cv2.imshow("kernel",kernel3)

eroded = cv2.erode(img,kernel3,iterations = 1)
#cv2.imshow("Erosion", eroded)


neg_img = np.zeros_like(img)
for i in range(len(img)):
    for j in range(len(img)):
        neg_img[i,j] = 255-img[i,j]

w = np.zeros([3,3],dtype = int)

for i in range(3):
    for j in range(3):
        w[i,j] = 1

w1 = (w) *255
w2 = np.uint8(w1)

rate = 50
w3 = cv2.resize(w2, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
#cv2.imshow("white",w3)
        
#kernel23 = np.zeros_like(w3,dtype = int)    
kernel23 = w3 - kernel3
#cv2.imshow("kernelw", kernel23)

eroded1 = cv2.erode(neg_img,kernel23,iterations = 1)
#cv2.imshow("Erosion1", eroded1)

main =  np.bitwise_and(eroded1, eroded)
cv2.imshow("main1", main)



m2 = np.array([[1, 1, 1], [0, 1, 0],[0,1,0]])
print(m2)

m21 = (m2) *255
m22 = np.uint8(m21)

rate = 50
m23 = cv2.resize(m22, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
#cv2.imshow("kernel2",m23)

ok = cv2.erode(img,m23,iterations = 1)
#cv2.imshow("Erosion21", ok)

n3 = w3 - m23
#cv2.imshow("kernelw", n3)

ok1 = cv2.erode(neg_img,n3,iterations = 1)
#cv2.imshow("Erosion22", ok1)

main1 =  np.bitwise_and(ok1, ok)
cv2.imshow("main2", main1)


p2 = np.array([[0, 1, 1], [0, 0, 1],[0,0,1]])
print(p2)

p21 = (p2) *255
p22 = np.uint8(p21)

rate = 50
p23 = cv2.resize(p22, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
#cv2.imshow("kernel2",p23)

ko = cv2.erode(img,p23,iterations = 1)
#cv2.imshow("Erosion21", ko)

l3 = w3 - p23
#cv2.imshow("kernelw", l3)

ko1 = cv2.erode(neg_img,n3,iterations = 1)
#cv2.imshow("Erosion22", ko1)

main34 =  np.bitwise_and(ko1, ko)
cv2.imshow("main3", main34)



cv2.waitKey(0)
cv2.destroyAllWindows()