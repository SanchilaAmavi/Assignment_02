import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("crop_field.png",0)

edges = cv.Canny(img,550,690)

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img,cmap='gray')

plt.subplot(1,2,2)
plt.title("Edge Image")
plt.imshow(edges,cmap='gray')

plt.show()