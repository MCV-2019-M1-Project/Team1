try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2

# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

def BFM(des1, des2, norm_type, max_distance_to_consider_match=1.0, cross_Check=False):
    '''
    Compares two local descriptors obtained with ORB(), obtains their matches and
        computes the mean correlation between matches
    Args:
        - desc1: descriptors obtained with ORB() of one image
        - desc2: descriptors obtained with ORB() of another image
        - max_distance_to_consider_match: if distance between two local descripts is
            smaller than max_distance_to_consider_match, the match will be discarded
        - norm_type: 
            NORM_L1, 
            NORM_L2, 
            NORM_HAMMING, 
            NORM_HAMMING2
            L1 and L2 norms are preferable choices for SIFT and SURF descriptors, 
            NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 
            should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description). 
    Returns
        - The number of matches between both descriptors that have a distance between
            them smaller than max_distance_to_consider_match
        - The mean correlation of all the matches between both descriptors that have a
            distance between them smaller than max_distance_to_consider_match
    '''
    if des1 is None or des2 is None:
        return 0
    else:
        bf = cv2.BFMatcher(norm_type, crossCheck=cross_Check)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = [match for match in matches if match.distance<=max_distance_to_consider_match]
        n_matches = len(matches)
        sum_distance = sum(match.distance for match in matches)
        if sum_distance == 0:
            mean_cor = 9999999
        else:
            mean_cor = n_matches / sum_distance
        return n_matches


def FLANN(des1, des2, descriptor='SURF'):
    if des1 is None or des2 is None:
        return 0, 0.0
    
    if descriptor not in ['SURF', 'ORB', 'SIFT']:
        raise NotImplementedError

    if descriptor == 'ORB':
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
    
    good_matches = []
    for matches_list in matches:
        if len(matches_list) == 2:
            m, n = matches_list
            if m.distance < 0.7*n.distance:
                good_matches.append(m)
        
    n_matches = len(good_matches)
    sum_distance = sum(match.distance for match in good_matches)
    if sum_distance == 0:
        mean_cor = 999999
    else:
        mean_cor = n_matches / sum_distance

    return n_matches