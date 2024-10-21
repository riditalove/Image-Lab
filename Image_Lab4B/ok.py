import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dpc
matplotlib.use('TkAgg')

point_list = []

img =cv2.imread('two_noise.jpeg',cv2.IMREAD_GRAYSCALE)

def onclick(event):
    global x,y
    ax = event.inaxes
    if ax is not None:
        x,y = ax.transData.inverted().transform([event.x,event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button,event.x,event.y,x,y))
        point_list.append((x,y))
        
def min_max(img):
    m = img.shape[0]
    n = img.shape[1]
    mini = np.min(img)
    maxi = np.max(img)
    for i in range(m):
        for j in range(n):
            img[i,j] = (((img[i,j]-mini)/(maxi-mini))*255)
    return np.array(img, dtype='uint8')

img_in = dpc(img)

#fourier transfor

ft = np.fft.fft2(img_in)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = abs(ft_shift)
magnitude_spectrum = np.log1p(abs(ft_shift))+1
magnitude_norm = min_max(magnitude_spectrum)

ang = np.angle(ft_shift)

plt.title("please select the seed values")
im = plt.imshow(img_in, cmap='gray')
im.figure.canvas.mpl_connect("button_press_event",onclick)
plt.show(block=True)

notch_filter=np.ones((img.shape),np.float32)
d0=int(input("Input the d0:"))
n=int(input("Input the n:"))
h=notch_filter.shape[0]//2
w=notch_filter.shape[1]//2
for u in range (0,notch_filter.shape[0]):
    for v in range (0,notch_filter.shape[1]):
        for k in range(0,len(point_list)):
            x=point_list[k][0]
            y=point_list[k][1]

            x,y=y,x
            if(x<=h):
                x2=h+(h-x)
            else:
                x2=h-(x-h)
            if (y <= w):
                y2 = w + (w - y)
            else:
                y2 = w - (y - w)
            d=pow(pow(u-x,2)+pow(v-y,2),n)
            d2 = pow(pow(u - x2, 2) + pow(v - y2, 2), n)
            if(d>d0):
                notch_filter[u][v]*=1
            else:
                notch_filter[u][v]*=0
            if (d2 > d0):
                notch_filter[u][v] *= 1
            else:
                notch_filter[u][v] *= 0


cv2.imshow("notch_filter.png",notch_filter)
notch_image2=cv2.imread('notch_filter.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('notch',notch_image2)


output = (magnitude_spectrum)*(notch_filter)
magnitude_spectrum_o = np.log1p(np.abs(output))+1
magnitude_o_norm = min_max(magnitude_spectrum_o)
plt.imsave("notch_filter_mul.png",magnitude_o_norm)
notch_image_mul2=cv2.imread('notch_filter_mul.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('notch2',notch_image_mul2)


final_result = np.multiply(output,np.exp(1j*ang))

img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back = min_max(img_back)


cv2.imshow("Inverse transform",img_back)



cv2.waitKey(0)
cv2.destroyAllWindows()