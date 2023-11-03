#!/usr/bin/env python
# coding: utf-8

# In[28]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

img = cv.imread('HeadCT.tif',0)

img = ((img/255)*200 + 30)
img = np.around(img).astype('uint8')
print(np.max(img))

plt.imshow(img, cmap = 'gray')
plt.show()

# gaussian filter
mean = 0
var = 100
sigma = var ** 0.5

image_gauss = np.zeros(img.shape, np.float32)
gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) 
image_gauss = img + gaussian
cv.normalize(image_gauss, image_gauss, 0, 255, cv.NORM_MINMAX, dtype=-1)

image_gauss = image_gauss.astype(np.uint8)

def add_noise(img):
 
    # Getting the dimensions of the image
    row , col = img.shape
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 1000 and 10000
    number_of_pixels = random.randint(1000, 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 1000 and 10000
    number_of_pixels = random.randint(1000 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img




cv.waitKey(0)
fig = plt.figure(figsize=(20,20))

plt.subplot(3, 2, 1)
plt.title('gaussian noise')
plt.imshow(image_gauss, cmap='gray')

s_p=add_noise(image_gauss)
plt.subplot(3, 2, 2)
plt.title('s_p noise')
plt.imshow(s_p, cmap='gray')

cv.imwrite('Retina.jpg',s_p)

median = cv.medianBlur(s_p,3)
plt.subplot(3, 2, 3)
plt.title('median')
plt.imshow(median, cmap='gray')

blur = cv.GaussianBlur(median,(5,5),0)
plt.subplot(3, 2, 4)
plt.title('blur')
plt.imshow(blur, cmap='gray')

lapkernel = np.array([[1, 1, 1],
                   [1, -8,1],
                   [1, 1, 1]])
mask = cv.filter2D(blur, -1, lapkernel)

blur = blur - (1)*mask

blur[np.where(blur < 0)] = 0
blur[np.where(blur > 255)] = 255

print(np.max(blur))
plt.subplot(3, 2, 5)
plt.title('laplacian')
plt.imshow(blur, cmap='gray')

plt.show()


# In[ ]:





# In[ ]:




