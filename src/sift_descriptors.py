import cv2
import matplotlib.pyplot as plt
from image_descriptors import similarity_for_descriptors

def SIFT_method(gray_img, kp=None, plot_results=False):
    """
    Computes SIFT descriptors and find keypoints if necessary.
    - Input:    gray_img: img where we want to find descriptors
                kp: keypoints (keypoint object) if were computed before with another method
    - Output:   Keypoints: the kp of the img
                descriptors: array of SIFT descriptors of the img -->size =(#keypoints, 128)
    """
    #Construct SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    if kp is not None:
        #if we have found the keypoints
        keypoints, descriptors = sift.compute(gray_img, kp) 
    else:
        #If we didn't find the keypoints
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        
    if plot_results:
        img = cv2.drawKeypoints(gray_img, keypoints, None)
        fig= plt.figure(figsize=(11,15))
        plt.imshow(img)
    return keypoints, descriptors


def SIFT_descriptors_matcher(des1, des2, norm_type=cv2.NORM_L2, thres_dist = 90):
    """
    Computes the macthes between the descriptors of two images 
    - des1, des2: descriptors of img 1 and img 2 
    - norm_type: norm type to compute the matches
    - thres_dist: threshold distance to select best matches
    
    Returns a list of matches
    """  
    #normType   One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. 
    #L1 and L2 norms are preferable choices for SIFT and SURF descriptors, 
    #NORM_HAMMING should be used with ORB, BRISK and BRIEF, 
    #NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4
        
    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort matches in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    good_matches = [m for m in matches if m.distance < thres_dist]  

    return good_matches



#path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P4\\qsd1_w4\\qsd1_w4\\00024.jpg'
#db_path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\Team1\\bbdd\\bbdd_00190.jpg'
#gray_q = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#gray_db = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
#
#kp1, desc1 = SIFT_method(gray_q)
#kp2, desc2 = SIFT_method(gray_db)
#
#match_rate = SIFT_descriptors_matcher(desc1, desc2, norm_type=cv2.NORM_L2)
#print(len(match_rate))