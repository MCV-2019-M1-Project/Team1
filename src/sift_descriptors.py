import cv2
import matplotlib.pyplot as plt


def SIFT_method(gray_img, kp=[], plot_results=False):
    """
    Computes SIFT descriptors and find keypoints if necessary.
    - Input:    gray_img: img where we want to find descriptors
                kp: keypoints (keypoint object) if were computed before with another method
    - Output:   Keypoints: the kp of the img
                descriptors: array of SIFT descriptors of the img -->size =(#keypoints, 128)
    """
    #Construct SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    if kp:
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

# def SIFT_descriptors_matcher(des1, des2):
#     ### BFMatcher with default params
#     bf = cv2.BFMatcher()
    
#     matches = bf.knnMatch(des1,des2,k=2)

#         # Apply ratio test
#     good = []
#     for m,n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append([m])
#     return good

#path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P4\\qsd1_w4\\qsd1_w4\\00024.jpg'
#db_path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\Team1\\bbdd\\bbdd_00190.jpg'
#gray_q = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#gray_db = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
#
#kp1, desc1 = SIFT_method(gray_q)
#kp2, desc2 = SIFT_method(gray_db)
#
#match_rate = SIFT_descriptors_matcher(desc1, desc2)
#print(len(match_rate))