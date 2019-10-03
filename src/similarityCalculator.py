import numpy as np
from scipy.spatial import distance
import cv2
#implement similarity measures to compare images

class similarityCalculator:  
    """
    To compute the similarity between query image histogram and the database images histograms
    
    """
    
    def __init__(self, query_features, db_features):
        
        self.query_features = query_features
        self.db_features = db_features
    
    def calc_euclidean_dist(self):
        euc_dist = distance.euclidean(self.query_features, self.db_features)
       
        return euc_dist
        
    def calc_L1_dist(self):
        # cityblock distance/L1 distance
        L1_dist = distance.cityblock(self.query_features, self.db_features)
        
        return L1_dist
    
    def calc_x2_dist(self):
        x2_dist = np.sum((self.query_features-self.db_features)**2/(self.query_features-self.db_features+1e-6))
        
        return x2_dist
        
    def calc_hist_intesection(self):
        #implementation of histograms intersection
        # minimum_val = np.sum(np.minimum(self.query_features,self.db_features))/np.sum(self.db_features)
        
        #cv compares histograms with intersection method        
        hist_intersection =cv2.CompareHist(self.query_features, self.db_features, cv2.HISTCMP_INTERSECT)
        
        return hist_intersection
        
    def calc_hellinger_kernel(self):
        #Synonym for Bhattacharyya distance 
        hellinger_dist = cv2.CompareHist(self.query_features, self.db_features, cv2.HISTCMP_HELLINGER)
        
        return hellinger_dist
        
    def calc_correlation(self):
        correlation = cv2.CompareHist(self.query_features, self.db_features, cv2.HISTCMP_CORREL)
        
        return correlation
    