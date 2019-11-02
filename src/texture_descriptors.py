from skimage import feature
from scipy.stats import itemfreq
from keypoint_detection import harris_corners
from image_descriptors import calc_blocks_from_image, calc_normalized_histogram
try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from distance import euclidean
import sys

def LBP(image, n_blocks=8, resize_level=2, P=8, R=2, method='uniform', histogram_size=[64], histogram_range=[0,255], mask=None):
    """
    Args:
        - image (M x N) array:
            An image in grayscale
        - n_blocks (int):
            divide the image into N x N blocks, where N is the n_blocks
        - resize_level (int):
            - 1 = 64x64
            - 2 = 128x128
            - 3 = 256x256
            - 4 = 512x512
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
        - histogram_size [int]: list with a single integer value, specify the size of the histogram
        - histogram_ranges [int, int]: list with a pair of ints, specify the range of the histogram
        - mask: matrix with the same same as input image.
        Returns:
            LBP histogram
    """

    if mask is not None:
        indices = np.where(mask != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            image = image[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
            mask = mask[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
        mask = mask.astype('uint8')
        image = cv2.bitwise_and(image, image, mask=mask)

    if resize_level not in [1,2,3,4]:
        raise NotImplementedError

    if resize_level == 1:
        resized_image = cv2.resize(image, (64,64))
    elif resize_level == 2:
        resized_image = cv2.resize(image, (128,128))
    elif resize_level == 3:
        resized_image = cv2.resize(image, (256,256))
    else:
        resized_image = cv2.resize(image, (512,512))

    image_blocks = calc_blocks_from_image(resized_image, n_blocks)

    image_histogram = []
    for image_block in image_blocks:
        block_lbp = np.float32(feature.local_binary_pattern(image_block, P, R, method))
        block_histogram = calc_normalized_histogram(block_lbp, None, [0], histogram_size, histogram_range)
        image_histogram.extend(block_histogram)

    return image_histogram

def HOG(image, mask=None, block_size=10, resize_level=2):
    """
    Args:
        - image (M x N) array:
            An image in grayscale
        - block_size (int):
            divide the image into N x N blocks, where N is the block_size
        - resize_level (int):
            - 1 = 64x64
            - 2 = 128x128
            - 3 = 256x256
            - 4 = 512x512
        - block_size (int, int) : specify the block size
    Returns
    """

    # Mask preprocessing
    if mask is not None:
        indices = np.where(mask != [0])
        if(indices[0].size != 0 and indices[1].size !=0):
            image = image[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
            mask = mask[min(indices[0]):max(indices[0]),min(indices[1]):max(indices[1])]
        mask = mask.astype('uint8')
        image = cv2.bitwise_and(image, image, mask=mask)

    if resize_level not in [1,2,3,4]:
        raise NotImplementedError

    if resize_level == 1:
        resized_image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
    elif resize_level == 2:
        resized_image = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)
    elif resize_level == 3:
        resized_image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    else:
        resized_image = cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)

    # Compute FOG feature
    fd = feature.hog(resized_image, orientations=8, pixels_per_cell=(block_size, block_size), cells_per_block=(5, 5), visualize=False, multichannel=True, feature_vector=True)

    return np.expand_dims(fd, axis=1).tolist()

def similarity_ORB(des1, des2, max_distance_to_consider_match=1.0):
    '''
    Compares two local descriptors obtained with ORB(), obtains their matches and
        computes the mean correlation between matches
    Args:
        - desc1: descriptors obtained with ORB() of one image
        - desc2: descriptors obtained with ORB() of another image
        - max_distance_to_consider_match: if distance between two local descripts is
            smaller than max_distance_to_consider_match, the match will be discarded
    Returns
        - The number of matches between both descriptors that have a distance between
            them smaller than max_distance_to_consider_match
        - The mean correlation of all the matches between both descriptors that have a
            distance between them smaller than max_distance_to_consider_match
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = [match for match in matches if match.distance<=max_distance_to_consider_match]
    n_matches = len(matches)
    sum_distance = sum(match.distance for match in matches)
    if sum_distance == 0:
        mean_cor = 999999
    else:
        mean_cor = n_matches / sum_distance
    return n_matches, mean_cor