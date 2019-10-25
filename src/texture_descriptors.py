from skimage import feature
from scipy.stats import itemfreq
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

def LBP(image, n_blocks, P, R, method='uniform'):
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

def HOG(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', 
        visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None):
    """
    Extract Histogram of Oriented Gradients (HOG) for a given image.
    Compute a Histogram of Oriented Gradients (HOG) by
        1. (optional) global image normalization
        2. computing the gradient image in row and col
        3. computing gradient histograms
        4. normalizing across blocks
        5. flattening into a feature vector
    Args:
        - image: Input image.
        - orientations: Number of orientation bins.
        - pixels_per_cell: 2-tuple (int, int), optional Size (in pixels) of a cell.
        - cells_per_block: 2-tuple (int, int), optional Number of cells in each block.
        - block_norm: Block normalization method:
            - L1 : Normalization using L1-norm.
            - L1-sqrt : Normalization using L1-norm, followed by square root.
            - L2 : Normalization using L2-norm.
            - L2-Hys : Normalization using L2-norm, followed by limiting the maximum 
            values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm. 
            (default) 
        - visualize: bool, optional
            Also return an image of the HOG. For each cell and orientation bin, the image 
            contains a line segment that is centered at the cell center, is perpendicular 
            to the midpoint of the range of angles spanned by the orientation bin, and has
             intensity proportional to the corresponding histogram value.
        - transform_sqrt: bool, optional
            Apply power law compression to normalize the image before processing. DO NOT 
            use this if the image contains negative values. Also see notes section below.  
        - feature_vector : bool, optional
            Return the data as a feature vector by calling .ravel() on the result just
            before returning.
        - multichannelboolean : optional
            If True, the last image dimension is considered as a color channel, 
            otherwise as spatial.
    Returns
    """

    hog = feature.hog(image)
    return hog

def DCT(image):    
    image_height, image_width = image.shape
    block_height = image_height / n_blocks
    block_width = image_width / n_blocks

    


img = cv2.imread('../bbdd/bbdd_00000.jpg', cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hog = HOG(img)
print(len(hog))