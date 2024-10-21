import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image


#img = cv2.imread("lena.jpg",0)
#cv2.imshow("original",img)
# Load the image
img = np.array(Image.open('lena.jpg').convert('L'))

# Define the kernel
kernel = np.array(([0,1,0],
              [1,-4,1],
              [0,1,0]),np.float32)

# Perform the convolution operation
result = cv2.filter2D(img, -1, kernel)

# Save the result as a new image
#Image.fromarray(result.astype('uint8')).save('result.png')
cv2.imshow("con",result)



cv2.waitKey(0)
cv2.destroyAllWindows()
