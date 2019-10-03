import numpy as np
from scipy.spatial import distance
import cv2

class Distance:
    
    def __init__(self, query_img, db_img):
        self.query_img = query_img
        self.db_img = db_img
#        self.db_img_filename = db_img_filename
    
    def method(self):
        
        return self.method
    
    def distance(self):
        
        return self.distance
    
    def  db_img_filename(self):
        
        return self.db_img.filename
    

    
    def calc_dist(self, similarity_method):
        """
        Calc the desired distance
        
        """
        query_hist = self.query_img.histogram
        
        
        
        db_hist = self.db_img.histogram
        
        
        if similarity_method == "euclidean":
            self.method = "euclidean"
            self.distance = distance.euclidean(query_hist, db_hist)
            return self.distance
        
        elif similarity_method == "L1_dist" :
            self.method = "L1_dist"
            self.distance  = distance.cityblock(query_hist, db_hist)
            return self.distance
        
        elif similarity_method == "x2_dist":
            self.method = "x2_dist"
            self.distance  = np.sum((query_hist-db_hist)**2/(query_hist-db_hist+1e-6))
            return self.distance
        
        elif similarity_method == "intersection":
            self.method = "intersection"
            self.distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_INTERSECT)
            return self.distance
        
        elif similarity_method == "hellinger":
            self.method = "hellinger"
            self.distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_HELLINGER)
            return self.distance
            
        elif similarity_method == "correlation":
            self.method = "correlation"
            self.distance = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CORREL)
            return self.distance