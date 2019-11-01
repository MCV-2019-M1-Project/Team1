try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2

import numpy as np
from skimage import feature

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
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,block_size,k_size,free_parameter)

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
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    keypoints = feature.blob_dog(gray, min_sigma, max_sigma, sigma_ratio, threshold, overlap)

    #Obtain keypoint mask
    mask_keypoints = np.zeros((gray.shape[0], gray.shape[1]))
    for keypoint in keypoints:
        mask_keypoints[keypoint[0], keypoint[1]] = 1
    if mask is not None:
        mask_keypoints = mask_keypoints * mask
    return mask_keypoints.astype(int)

'''
#Example of usage

im_bd = cv2.imread('/home/mvila/Documents/MasterCV/M1/project/codi/Team1/bbdd/bbdd_00023.jpg')
mask = np.ones((im_bd.shape[0], im_bd.shape[1]))
m = difference_of_gaussians(im_bd, mask=mask, threshold=0.5)
m = np.float32(m)
cv2.imshow('or', m)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''
