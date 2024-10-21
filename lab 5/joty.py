
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

point_list = []
img = cv2.imread("input.jpg", 0)
# img = np.zeros((10,12), np.uint8)
# img[4:6, 5:7] = 1
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
        print(
            "button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (event.button, event.x, event.y, x, y)
        )
        point_list.append((x, y))


X = np.zeros_like(img)
plt.title("Please select seed pixel from the input")
im = plt.imshow(img, cmap="gray")
im.figure.canvas.mpl_connect("button_press_event", onclick)
plt.show(block=True)

print(point_list)


r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)  # _INV)
cv2.imshow("Nothing", img)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel = kernel

x = point_list[0][0]
y = point_list[0][1]
x0 = np.zeros_like(img)
x0[x][y] = 1
imgg = np.zeros_like(img)


while 1:
    temp = cv2.dilate(x0, kernel, iterations=1)
    neg_imgg = 1 - img
    x1 = np.bitwise_and(temp, neg_imgg)
    if (x0 == x1).all():
        break
    x0 = x1

x1 = x1 * 255
imgg = np.bitwise_or(x1, img)
cv2.imshow("Original", imgg)

cv2.waitKey(0)
cv2.destroyAllWindows()
