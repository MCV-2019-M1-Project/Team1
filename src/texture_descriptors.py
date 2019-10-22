from skimage import feature
from scipy.stats import itemfreq
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def LBP(image, P, R, method='uniform'):
    """
    Args:
        - image: Gray scale image (N x M) array
        - P: Number of circularly symmetric neighbour set points 
             (quantization of the angular space).
        - R: Radius of circle (spatial resolution of the operator).
        - method: Method to determine the pattern.
            ‘default’: original local binary pattern which is gray scale but not
                rotation invariant.

            ‘ror’: extension of default implementation which is gray scale and
                rotation invariant.

            ‘uniform’: improved rotation invariance with uniform patterns and
                finer quantization of the angular space which is gray scale and rotation invariant.

            ‘nri_uniform’: non rotation-invariant uniform patterns variant
                which is only gray scale invariant [2].

            ‘var’: rotation invariant variance measures of the contrast of local
                image texture which is rotation but not gray scale invariant.
        Returns:
            LBP image (N x M) array
    """

    lbp = feature.local_binary_pattern(image, P, R, method)
    x = itemfreq(lbp.ravel())
    hist = x[:, 1]/sum(x[:, 1])
    return hist



im = cv2.imread('../bbdd/bbdd_00000.jpg')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
hist = LBP(im_gray, 24, 8)
print(hist)