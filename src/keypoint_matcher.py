try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2
import sys

# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

def BFM_KNN(des1, des2, norm_type):
    if des1 is None or des2 is None:
        return 0
       
    matches = cv2.BFMatcher(norm_type, crossCheck=False).knnMatch(des1, des2, k=2)   

    good_matches = 0   
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7*n.distance:
                good_matches += 1
    return good_matches

def FLANN_KNN(des1, des2, descriptor='SURF'):
    if des1 is None or des2 is None:
        return 0

    
    if descriptor.upper() not in ['SURF', 'ORB', 'SIFT']:
        raise NotImplementedError

    if descriptor.upper() == 'ORB':
        index_params = dict(algorithm = 6, 
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
    else:
        index_params = dict(algorithm = 1,
                            trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good_matches = 0   
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7*n.distance:
                good_matches += 1
    return good_matches