import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from copy import deepcopy as dpc

point_list=[]

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
        point_list.append((x,y))


X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

print(point_list)

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')

# take input
img_input = cv2.imread('two_noise.jpeg',cv2.IMREAD_GRAYSCALE)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)


notch = np.zeros((img.shape),np.float32)
d = int(input("Select a radius: "))
n = int(input("Select a values: "))
h = img.shape[0]//2
w = img.shape[1]//2

for i in range(0, notch.shape[0]):
    for j in range(0, notch.shape[1]):
        a=0
        b=0
        c=0
        for k in range(0,len(point_list)):
            x = point_list[k][0]
            y = point_list[k][1]
            x,y = y,x
            a = pow((pow((notch[i,j]-h-x),2)+pow((notch[i-j]-w-y),2)),0.5)+a
            b = pow((pow((notch[i,j]-h+x),2)+pow((notch[i-j]-w+y),2)),0.5)+b
            
        notch[i][j] = (1/(1+(pow((d/a),n))))+(1/(1+(pow((d/b),n))))+c
        
plt.imsave("notch_filter.png",notch)
notch_image2=cv2.imread('notch.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('notch',notch_image2)



output=magnitude_spectrum_ac*notch
magnitude_spectrum_o = np.log1p(np.abs(output)+1)
magnitude_spectrum_scaled_o = min_max_normalize(magnitude_spectrum_o)
plt.imsave("notch_filter_mul.png",magnitude_spectrum_scaled_o)
notch_image_mul2=cv2.imread('notch_filter_mul.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('notch2',notch_image_mul2)            
            
                       
final_result = np.multiply(output, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)


cv2.imshow("Inverse transform",img_back_scaled)





cv2.waitKey(0)


## plot
cv2.imshow("input", img_input)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("Inverse transform",img_back_scaled)



cv2.waitKey(0)
cv2.destroyAllWindows() 

