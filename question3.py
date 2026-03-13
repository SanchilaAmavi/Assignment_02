import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("crop_field.png",0)
edges = cv.Canny(img,550,690)
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]
m, b = np.polyfit(x, y, 1)
y_fit = m * x + b

plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x, y_fit, color='red', linewidth=2, label="Least Squares Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Least Squares Line Fitting")
plt.legend()

plt.show()