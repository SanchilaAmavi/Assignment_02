import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
img = cv.imread("crop_field.png")
if img is None:
    print("Error: Image not found!")
    exit()
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
edges = cv.Canny(img, 550, 690)
indices = np.where(edges != 0)
x = indices[1]
y = indices[0]
x_r = x.reshape(-1, 1)
ransac = RANSACRegressor()
ransac.fit(x_r, y)
y_ransac = ransac.predict(x_r)
plt.figure(figsize=(8,6))
plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x, y_ransac, color='orange', linewidth=2, label="RANSAC Line")
plt.gca().invert_yaxis()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Line Estimation using RANSAC")
plt.legend()
plt.show()