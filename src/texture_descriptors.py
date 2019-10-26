from skimage import feature
from scipy.stats import itemfreq
from image_descriptors import calc_blocks_from_image, calc_normalized_histogram
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def LBP(image, n_blocks, P, R, method='uniform', mask=None, histogram_size=[255], histogram_ranges=[0,255]):
    """
    Args:
        - image: Gray scale image (N x M) array
        - P (int): Number of circularly symmetric neighbour set points 
             (quantization of the angular space).
        - R (int): Radius of circle (spatial resolution of the operator).
        - method (string): Method to determine the pattern.
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
        - mask: matrix with the same same as input image. By default will use a matrix of one's with
                the same input image size. Should be np.uint8 TYPE!
        - histogram_size [int]: list with a single integer value, specify the size of the histogram
        - histogram_ranges [int, int]: list with a pair of ints, specify the range of the histogram
        Returns:
            LBP histogram
    """
    def LBP(image, mask, P, R, method, hist_size, ranges):
        # An example how to uyse the local binary pattern
        # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html

        # Convert the output into an array of float 32
        lbp = np.float32(feature.local_binary_pattern(image, P, R, method))
        
        # Calc the normalized histogram of the LBP feature
        return calc_normalized_histogram(image, mask, [0], hist_size, ranges)
    
    # Get the image blocks
    image_blocks = calc_blocks_from_image(image, n_blocks)

    if mask is None:
        mask = np.ones(image.shape, np.uint8)

    # Get the mask blocks
    mask_blocks = calc_blocks_from_image(mask, n_blocks)
    
    image_histogram = None
    for image_block, mask_block in zip(image_blocks, mask_blocks):
        block_histogram = LBP(image_block, mask_block, P, R, method, histogram_size, histogram_ranges)
        if image_histogram is not None:
            image_histogram = np.concatenate((block_histogram, image_histogram))
        else:
            image_histogram = block_histogram
    
    return image_histogram

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

def DCT(image, resize_window, n_blocks, k_coeff):
    """
    Calcs the DCT feature of a given image. That image should be grayscale. There's two
    parameters: resize_Window and n_blocks. The function will apply a resize on that image
    and then divide it into blocks.

    Important:
        The block size must be non odd. e.g 20x20. Otherwise an error will be raised warns you
        that the resize window and n_blocks do not outputs non odd blocks.
    Args:
        - image : grayscale image
        - resize_window (int, int): specify wich resize will be applied on the image
        - n_blocks (int): number of blocks
        - k_coeff (int): take the K highest coeff from each block
    Returns:
        A list of the best K_coeff of each block, ordered by a ZigZag method. i.e, ordered DESC by energy
    """

    # DCT do not accepts Odd-size, it throws the following errors:
    # Odd-size DCT's are not implemented in function 'apply'
    # Then we should resize the image into non Odd size    
    # Resize the image into none odd
    resized_image = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)    

    # Get the image blocks
    image_blocks = calc_blocks_from_image(resized_image, n_blocks)

    # Store the DCT feature
    feature = []
    for image_block in image_blocks:
        # Normalize all the values
        # Otherwsie: (-215:Assertion failed) type == CV_32FC1 || type == CV_64FC1 in function 'dct'
        image_block = np.float32(image_block/255.0)

        try:
            dct = cv2.dct(image_block)

            # Why apply ZigZag:
            # https://stackoverflow.com/questions/52193536/image-compressing-zigzag-after-discrete-cosine-transform
            # https://en.wikipedia.org/wiki/JPEG#Discrete_cosine_transform
            # Code found here: https://stackoverflow.com/questions/39440633/matrix-to-vector-with-python-numpy
            flatt_dct = np.concatenate([np.diagonal(dct[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-dct.shape[0], dct.shape[0])])

            # Take the k Coeff
            coeff = flatt_dct[:k_coeff]

            # Store'm
            feature.extend(coeff)
        except cv2.error:
            print('Error: during the calc of image block DCT. The image block has the following dim: {} and should be a non ODD size!'.format(image_block.shape))
            sys.exit()
        
    return feature