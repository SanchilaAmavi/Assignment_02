import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("crop_field.png",0)

edges = cv.Canny(img,550,690)

indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

plt.scatter(x,y,s=1)
plt.gca().invert_yaxis()
plt.title("Scatter Plot of Edge Points")
plt.xlabel("x")
plt.ylabel("y")

plt.show()