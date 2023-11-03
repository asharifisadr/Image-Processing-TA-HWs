import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('MRI-Head.png', 0)
ra = 50
img = img.astype('float32') * (1-2*ra/255) + ra
vid = cv.VideoWriter('MRI-Head.avi', cv.VideoWriter_fourcc(*'XVID'), 5.0, img.shape[::-1], 0)
frames = 20

for i in range(frames):
    img_ = np.clip(img + np.random.randn(*img.shape) * 0.75 * ra, 0, 255).astype('uint8')
    vid.write(img_)

vid.release()