from cv2 import cv2

# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

def BFM(des1, des2, max_distance_to_consider_match=1.0):
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


def FLANN(des1, des2, FLANN_INDEX_KDTREE=1, trees=5, checks=50, K_MATCHES=2, max_distance_to_consider_match=1.0):    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,K_MATCHES=2)
    
    # Apply ratio test SEE THE TUTORIAL ON THE TOP OF THIS FILE TO SEE WHAT IT DO
    good_matches = []
    for m,n in matches:
        if m.distance < max_distance_to_consider_match*n.distance:
            good_matches.append(m)
        
    n_matches = len(good_matches)
    sum_distance = sum(match.distance for match in good_matches)
    if sum_distance == 0:
        mean_cor = 999999
    else:
        mean_cor = n_matches / sum_distance
    return n_matches, mean_cor