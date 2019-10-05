import numpy as np
from scipy.spatial import distance
from cv2 import cv2


class Distance:
    def __init__(self, query_img, db_img):
        self.query_img = query_img
        self.db_img = db_img
        
    def method(self):

        return self._method

    def distance(self):

        return self._distance

    def db_img_filename(self):

        return self.db_img.filename
    
    def maximization(self):
        """
        True if method requires maximization, False if minimization
        """
        
        return self._maximization
        
    
    def calc_dist(self, similarity_method):
        """
        Calc the desired distance
        
        """
        query_hist = self.query_img.histogram

        db_hist = self.db_img.histogram

        if similarity_method == "euclidean":
            self._method = "euclidean"
            self._distance = distance.euclidean(query_hist, db_hist)
            self._maximization = False
            return self._distance

        elif similarity_method == "L1_dist":
            self._method = "L1_dist"
            self._distance  = distance.cityblock(query_hist, db_hist)
            self._maximization = False
            return self._distance

        elif similarity_method == "x2_dist":
            self._method = "x2_dist"
            self._distance  = np.sum((query_hist-db_hist)**2/(query_hist-db_hist+1e-6))
            self._maximization = False
            return self._distance

        elif similarity_method == "intersection":
            self._method = "intersection"
            self._distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_INTERSECT)
            self._maximization = True
            return self._distance

        elif similarity_method == "hellinger":
            self._method = "hellinger"
            self._distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_HELLINGER)
            self._maximization = False
            return self._distance

        elif similarity_method == "correlation":
            self._method = "correlation"
            self._distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CORREL)
            self._maximization = True
            return self._distance