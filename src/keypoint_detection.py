try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2

import numpy as np
from skimage import feature
import math

def harris_corners(image, mask = None, block_size=2, k_size=3, free_parameter=0.04, threshold=0.01):
    """
    Obtain the harris corners of the image
    Args:
        - image: image in BGR color space
        - mask: None or mask with 0's in the pixels that don't have to be taken into account and 1's in the other positions
        - block_size: neighborhood size (for each pixel value block_size*block_size neighbourhood is considered)
        - k_size: Aperture parameter for the Sobel() operator
        - free_parameter: Harris detector free parameter
        - threshold: threshold for discarding small valued corners
    Returns
        Mask with all the corners set at 1 and the rest at 0
    """


    image = np.float32(image)
    dst = cv2.cornerHarris(image,block_size,k_size,free_parameter)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    mask_corners = dst>threshold*dst.max()
    if mask is not None:
        mask_corners = mask_corners * mask
    return mask_corners.astype(int)

def difference_of_gaussians(image, ):
    blobs = feature.blob_dog(image, threshold=0.1, min_sigma=1, max_sigma=30, sigma_ratio=1.6, overlap=0.5)
    keypoints = []
    for blob in blobs:
        x, y, r = blob
        keypoint = cv2.KeyPoint(x, y, r*np.sqrt(2) * 2)
        keypoints.append(keypoint)
    return keypoints

def determinant_of_hessian(image):
    blobs = feature.blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.001, overlap=0.5, log_scale=False)
    keypoints = []
    for blob in blobs:
        x, y, r = blob
        keypoint = cv2.KeyPoint(x, y, r*np.sqrt(2) * 2)
        keypoints.append(keypoint)
    return keypoints

def laplacian_of_gaussian(image):
    blobs = feature.blob_log(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1, overlap=0.5, log_scale=False, exclude_border=False)
    keypoints = []
    for blob in blobs:
        x, y, r = blob
        keypoint = cv2.KeyPoint(x, y, r*np.sqrt(2) * 2)
        keypoints.append(keypoint)
    return keypoints

def ORB(image):
    orb = cv2.ORB_create()
    return orb.detect(image)

def SIFT(image):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detect(image)

def SURF(image, hessian=400):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian)
    return surf.detect(image)