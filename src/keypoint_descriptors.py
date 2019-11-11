try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2

def SURF_descriptor(gray_img, keypoints):
    surf = cv2.xfeatures2d.SURF_create()
    _, descriptors = surf.compute(gray_img, keypoints)
    return descriptors

def ORB_descriptor(image, keypoints):    
    orb = cv2.ORB_create()
    _, des = orb.compute(image, keypoints)
    return des

def SIFT_descriptor(gray_img, keypoints):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 500, nOctaveLayers=4, edgeThreshold=10, sigma=1.4)
    _, descriptors = sift.compute(gray_img, keypoints)
    return descriptors