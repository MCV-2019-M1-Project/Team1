from similarityCalculator import similarityCalculator

class Distance:
    
    def __init__(self, query_img, db_img):
        self.query_img = query_img
        self.db_img = db_img
    
    def euclidean(self):
        
        return self.euclidean
    
    def L1(self):
    
        return self.L1
    
    def x2(self):
    
        return self.x2
    
    def intersection(self):
    
        return self.intersection
    
    def hellinger(self):
    
        return self.hellinger
    
    def correlation(self):
    
        return self.correlation
    
    def calc_dist(self, query_hist, db_hist, num_similarities, similarity_methods):
        """
        Calc the desired distance
        
        """
        if "euclidean" in similarity_methods:
            self.euclidean = similarityCalculator.calc_euclidean_dist(query_hist, db_hist)
        
        if "L1_dist" in similarity_methods:
            self.L1  = similarityCalculator.calc_L1_dist(query_hist, db_hist)
        
        if "x2_dist" in similarity_methods:
            self.x2  = similarityCalculator.calc_x2_dist(query_hist, db_hist)
        
        if "intersection" in similarity_methods:
            self.intersection = similarityCalculator.calc_hist_intesection(query_hist, db_hist)
        
        if "hellinger" in similarity_methods:
            self.hellinger = similarityCalculator.calc_hellinger_kernel(query_hist, db_hist)
            
        if "correlation" in similarity_methods:
            self.correlation = similarityCalculator.calc_correlation(query_hist, db_hist)