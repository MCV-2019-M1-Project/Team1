try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2

import numpy as np

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
