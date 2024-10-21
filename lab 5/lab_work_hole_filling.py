import math
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
point_list = []
x = None
y = None


def min_max_normalization(image):
    h=image.shape[0]
    w=image.shape[1]
    min=np.min(image)
    max=np.max(image)
    output=np.zeros((image.shape),np.uint8)

    for i in range(0,h):
        for j in range(0,w):
            temp=((image[i][j]-min)/(max-min))*255
            output[i][j]=temp
    return output


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

image=cv2.imread('input.jpg',cv2.IMREAD_GRAYSCALE)


print(np.min(image),np.max(image))

image2=image//255
image_inv=1-image2
image_inv*=255


im = plt.imshow(image, cmap='gray')
im.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)
print(point_list)

se=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
print(se)
se1=(se)*255
se1=np.uint8(se1)
rate = 50
kernel1 = cv2.resize(se1, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
# plt.imshow(kernel1,'gray')
# plt.show()
cv2.imshow('kernel',kernel1)

output=np.zeros((image.shape),np.float32)
for i in range(0,len(point_list)):
    x=point_list[i][0]
    y=point_list[i][1]
    x,y=y,x
    print(x,y)
    z=np.zeros((image.shape),np.uint8)
    z[x][y]=1

    while(1):
        z1=z

        z=cv2.dilate(z,se,iterations=1)

        z=cv2.bitwise_and(image_inv,z)
        # plt.imshow(z,'gray')
        # plt.show()
        if(np.array_equal(z1,z)):
            break

    plt.imshow(z,'gray')
    plt.show()
    z=z*255

    # output = image + z
    # plt.imshow(output, 'gray')
    # plt.show()
    output+=z

output=min_max_normalization(output)
output = image + output
output=min_max_normalization(output)
plt.imshow(output, 'gray')
plt.show()



cv2.waitKey()
cv2.destroyAllWindows()