from cv2 import cv2
from keypoint_detection import (
    determinant_of_hessian
)

def _create_keypoint_mask(keypoints, keypoint_diameter):
    kp = []
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            if keypoints[i,j] == 1:
                kp.append(cv2.KeyPoint(x=i, y=j, _size=keypoint_diameter))
    return kp


def surf_descriptor(image, keypoints_mask=None, keypoint_diameter=7):
    """
    Extract descriptors from image using the SURF method.
    Args:
        image (M x N):
            a grayscale image
        keypoints (M x N):
            image mask
    Returns:
        list of descriptors objects
    """

    # https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html
    surf = cv2.xfeatures2d.SURF_create()
    if keypoints_mask is None:
        keypoints = surf.detect(image, None)
    else:
        keypoints = _create_keypoint_mask(keypoints_mask, keypoint_diameter)

    _, descriptors = surf.compute(image, keypoints)
    return descriptors

def ORB(image, keypoint_mask=None, keypoint_diameter=7):
    """
    Obtains the orb descriptors of the keypoints in image
    Args:
        - image: image in BGR color space
        - mask: None or mask with all the keypoints set at 1 and the rest at 0
        - keypoint_diameter: diameter of the keypoint that will be used to compute the descriptor
    Returns
        Orb descriptors for given or found keypoints
    """
    orb = cv2.ORB_create()
    #Obtain keypoints
    if keypoint_mask is None:
        kp = orb.detect(image,None)
    else:
        kp = _create_keypoint_mask(keypoint_mask, keypoint_diameter)
    #Obtain descriptors
    kp, des = orb.compute(image, kp)
    return des