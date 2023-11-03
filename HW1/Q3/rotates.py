import numpy as np
def rotate_forward(image, ang_rad):
    'Rotate image with angle of ang_rad.'
    forward_image = np.zeros(image.shape)
    r, c = image.shape
    i, j = np.indices(image.shape)
    i_ = np.int32(np.round_(i*np.cos(ang_rad)-j*np.sin(ang_rad)))
    j_ = np.int32(np.round_(i*np.sin(ang_rad)+j*np.cos(ang_rad)))
    mask = (0<=i_) & (i_<r) & (0<=j_) & (j_<c)
    forward_image[i_[mask], j_[mask]] = image[i[mask], j[mask]]
    return forward_image