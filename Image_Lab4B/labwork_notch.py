# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
matplotlib.use('TkAgg')

point_list = []

img =cv2.imread('two_noise.jpeg',cv2.IMREAD_GRAYSCALE)

# click and seed point set up
x = None
y = None


# The mouse coordinate system and the Matplotlib coordinate system are different, handle that
def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, x, y))
        point_list.append((x, y))


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('two_noise.jpeg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac=np.abs(ft_shift)
magnitude_spectrum = np.log1p(np.abs(ft_shift)+1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

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



output=magnitude_spectrum_ac*notch_filter
magnitude_spectrum_o = np.log1p(np.abs(output)+1)
magnitude_spectrum_scaled_o = min_max_normalize(magnitude_spectrum_o)
plt.imsave("notch_filter_mul.png",magnitude_spectrum_scaled_o)
notch_image_mul2=cv2.imread('notch_filter_mul.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('notch2',notch_image_mul2)




## phase add
final_result = np.multiply(output, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)


cv2.imshow("Inverse transform",img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
