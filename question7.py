import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("crop_field.png", 0)

if img is None:
    print("Image not found")
    exit()

edges = cv.Canny(img, 550, 690)
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]
plt.scatter(x, y, s=1, label="Edge Points")
points = np.column_stack((x, y))
centroid = np.mean(points, axis=0)
points_centered = points - centroid
U, S, Vt = np.linalg.svd(points_centered)
direction = Vt[0]
t = np.linspace(-2000, 2000, 1000)
line_x = centroid[0] + direction[0] * t
line_y = centroid[1] + direction[1] * t
plt.plot(line_x, line_y, 'r', linewidth=2, label="Total Least Squares Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Total Least Squares Line Fit")
plt.legend()

m_tls = direction[1] / direction[0]
theta_tls = np.degrees(np.arctan(m_tls))

print("Estimated crop field angle (TLS) =", theta_tls, "degrees")

plt.show()