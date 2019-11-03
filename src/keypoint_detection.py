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

def difference_of_gaussians(image, mask = None, threshold=0.5, min_sigma=1, max_sigma=50, sigma_ratio=1.6, overlap=0.5):
    """
    Obtain keypoints of the input image using the difference_of_gaussians method
    Args:
        - image: image in BGR color space
        - mask: None or mask with 0's in the pixels that don't have to be taken into account and 1's in the other positions
        - threshold: threshold for discarding small valued corners
        - min_sigma: scalar or sequence of scalars, the minimum standard deviation for Gaussian kernel
        - max_sigma: scalar or sequence of scalars, the maximum standard deviation for Gaussian kernel
        - sigma_ratio: float, the ratio between the standard deviation of Gaussian Kernels used for computing the Difference of Gaussians
        - threshold: float, the absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.
        - overlap: float, a value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated.
    Returns
        Mask with all the keypoints set at 1 and the rest at 0
    """

    keypoints = feature.blob_dog(image, min_sigma, max_sigma, sigma_ratio, threshold, overlap)

    #Obtain keypoint mask
    mask_keypoints = np.zeros((image.shape[0], image.shape[1]))
    for keypoint in keypoints:
        mask_keypoints[int(keypoint[0]), int(keypoint[1])] = 1
    if mask is not None:
        mask_keypoints = mask_keypoints * mask
    return mask_keypoints.astype(int)

def determinant_of_hessian(image, mask = None, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False):
    """
    mage2D ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.

    min_sigmafloat, optional
        The minimum standard deviation for Gaussian Kernel used to compute Hessian matrix. Keep this low to detect smaller blobs.

    max_sigmafloat, optional
        The maximum standard deviation for Gaussian Kernel used to compute Hessian matrix. Keep this high to detect larger blobs.

    num_sigmaint, optional
        The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.

    thresholdfloat, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect less prominent blobs.

    overlapfloat, optional
        A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated.

    log_scalebool, optional
        If set intermediate values of standard deviations are interpolated using a logarithmic scale to the base 10. If not, linear interpolation is used.
    """

    # Example https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py
    blobs_doh = feature.blob_doh(image, min_sigma, max_sigma, num_sigma, threshold, overlap, log_scale)

    # generate a keypoint mask
    mask_keypoints = np.zeros((image.shape[0], image.shape[1]))
    for keypoint in blobs_doh:
        mask_keypoints[int(keypoint[0]), int(keypoint[1])] = 1
    if mask is not None:
        mask_keypoints = mask_keypoints * mask
    return mask_keypoints.astype(int)