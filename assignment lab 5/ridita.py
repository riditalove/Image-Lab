import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import math
from copy import deepcopy as dpc

input_img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

def gauss_noise(img,sigma):
    kernel = np.zeros_like(img,np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cons = 2*(math.pi)*sigma*sigma
            k = -((i*i) + (j*j))
            c = 2*(sigma**2)
            kernel[i,j] = (math.exp(k/c))/cons
    kernel = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX)
    kernel = kernel.astype(np.uint8)
    return kernel

s1=int(input("Input the sigma 1:"))
s2=int(input("Input the sigma 2:"))

noise = np.zeros_like(input_img,np.float32)
noise = gauss_noise(input_img, s1)
noise1 = gauss_noise(input_img,s2)
noise1 = np.flip(noise1,0)
noise1 = np.flip(noise1,1)
two_noise = np.zeros_like(input_img,np.float32)
two_noise =  noise + noise1
noisy_img = np.zeros_like(input_img,np.float32)
noisy_img = cv2.add(two_noise,input_img)


def log(img):
    log_img = np.zeros_like(img,np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            log_img[i,j]  =  np.log1p(img[i,j])
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    log_img = log_img.astype(np.uint8)
    return log_img


def inv_log(img):
    inv_log_img = np.zeros_like(img,np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = 255/(np.log(1+255))
            inv_log_img[i,j]  =  np.exp(img[i,j]**1/c)-1 
    inv_log_img = np.array(inv_log_img, dtype = np.uint8)
    return inv_log_img


def Homomorphic_filter(img):
    yh = 1.8
    yl = 0.1
    c = 5
    d0 = 10 
    m = img.shape[0]
    n = img.shape[1]
    filter = np.zeros_like(img,np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dp = ((i-(m//2))**2) + ((j-(n//2))**2)
            dk = np.sqrt(dp)
            p = (dk**2)/(d0**2)
            q = (-1)*c*p
            r = 1 - np.exp(q)
            filter[i,j] = ((yh-yl)*r) + yl
    return filter
            

log = log(noisy_img)

homo_filter = Homomorphic_filter(input_img)            


#inv_log = inv_log(log)
cv2.imshow("Noise imaage",noisy_img)
cv2.imshow("Just noise",two_noise)
cv2.imshow("homo", homo_filter)


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range (img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j]-inp_min)/(inp_max-inp_min))*255)
    return np.array(img_inp, dtype='uint8')


real_img = dpc(log)

image_size = real_img.shape[0] * real_img.shape[1]

# fourier transform
ft = np.fft.fft2(real_img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

#mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

## phase add
final_result = np.multiply(magnitude_spectrum_ac*homo_filter, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("Magnitude Spectrum",magnitude_spectrum_scaled)

cv2.imshow("Inverse transform",img_back_scaled)

#inverse_log
inv_log = inv_log(img_back_scaled)

cv2.imshow("logged image",real_img)
cv2.imshow("Input", input_img)
#cv2.imshow("Output",inv_log)
#cv2.imshow("Input", final_result)

cv2.waitKey(0)
cv2.destroyAllWindows()