from skimage import feature
from scipy.stats import itemfreq
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def multiblock_LBP(image, n_blocks, P, R, method='uniform'):
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
            LBP histogram
    """
    def LBP(image, P, R, method):
        lbp = feature.local_binary_pattern(image, P, R, method)    
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist
    
    image_height, image_width = image.shape
    block_height = image_height / n_blocks
    block_width = image_width / n_blocks

    blocks = []
    for y in range(n_blocks):
        for x in range(n_blocks):
            block = image[int(y*block_height):int((y+1)*block_height), int(x*block_width):int((x+1)*block_width)]
            blocks.append(block)
    
    block_feature = []
    for block in blocks:
        block_histogram = LBP(block, P, R, method)
        block_feature.extend(block_histogram)
    
    return block_feature