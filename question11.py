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
X = x.reshape(-1,1)
ransac = RANSACRegressor(LinearRegression())
ransac.fit(X, y)
m = ransac.estimator_.coef_[0]
b = ransac.estimator_.intercept_
x_line = np.linspace(min(x), max(x), 1000)
y_line = m * x_line + b
theta = np.degrees(np.arctan(m))
print("Estimated crop field angle (Proposed Algorithm - RANSAC) =", theta, "degrees")
plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x_line, y_line, color='green', linewidth=2, label="RANSAC Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Crop Field Direction Estimation using RANSAC")
plt.legend()
plt.show()