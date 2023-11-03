import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from rotate import rotate_forward

img = cv.imread("transformed.png", 0)

plt.imshow(rotate_forward(img, np.pi/12), 'gray')
plt.show()