import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

img = cv.imread("crop_field.png", 0)

if img is None:
    print("Image not found")
    exit()

edges = cv.Canny(img, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

X = x.reshape(-1, 1)

ransac = RANSACRegressor(LinearRegression())
ransac.fit(X, y)

slope_r = ransac.estimator_.coef_[0]
intercept_r = ransac.estimator_.intercept_

theta_r = np.degrees(np.arctan(slope_r))

print("RANSAC angle =", theta_r)

x_line = np.linspace(min(x), max(x), 1000)
y_line = slope_r * x_line + intercept_r

plt.figure(figsize=(8,6))
plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x_line, y_line, color='green', linewidth=2, label="RANSAC Line")

plt.gca().invert_yaxis()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Crop Field Direction Estimation using RANSAC")
plt.legend()
plt.show()