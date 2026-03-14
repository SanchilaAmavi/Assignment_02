import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("crop_field.png")   
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]


data = np.vstack((x, y)).T
mean = np.mean(data, axis=0)

U, S, Vt = np.linalg.svd(data - mean)

direction = Vt[0]

slope_tls = direction[1] / direction[0]

y_tls = slope_tls * (x - mean[0]) + mean[1]

plt.figure(figsize=(8,6))
plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x, y_tls, color='green', linewidth=2, label="Total Least Squares Line")

plt.gca().invert_yaxis() 
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Total Least Squares Fit")
plt.legend()

plt.show()

print("TLS Slope:", slope_tls)