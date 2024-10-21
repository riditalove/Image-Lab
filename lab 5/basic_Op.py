import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


img = cv2.imread('input.jpg', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)#_INV)
cv2.imshow("Original", img)

point_list=[]
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x,y))

im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #cv2.MORPH_RECT for all 1s
kernel1 = (kernel1) *255
kernel = np.uint8(kernel1)

rate = 50
kernel1 = cv2.resize(kernel, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("kernel",kernel1)



demo = np.zeros((img.shape[0],img.shape[1]), np.uint8)
demo[point_list[0][0]][point_list[0][1]] = 255



kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
#kernel = np.ones((25,25),np.uint8)

neg_imgg = 1-img
x0 = demo

while 1:
        
    temp = cv2.dilate(x0, kernel, iterations = 1)
    x1 = np.bitwise_and(temp, neg_imgg) 
    plt.imshow(x1, 'gray')
    plt.show()
    if(np.sum(x1) == np.sum(x0)):
        break
    x0 = x1

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(x1[i][j] ==1):
            img[i][j] = x1[i][j]*255

out = np.bitwise_or(x1, img)
dilated = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow("Dilation", out)


eroded = cv2.erode(img,kernel,iterations = 1)
#cv2.imshow("Erosion", eroded)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 1)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations =1)
#cv2.imshow("Closing", closed)
plt.imshow(out, 'gray')
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()