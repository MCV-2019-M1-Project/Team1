from similarityCalculator import similarityCalculator

class Distance:
    
    def __init__(self, query_img, db_img):
        self.query_img = query_img
        self.db_img = db_img
    
    def method(self):
        
        return self.method
    
    def distance(self):
        
        return self.distance
    

    
    def calc_dist(self, query_hist, db_hist, num_similarities, similarity_methods):
        """
        Calc the desired distance
        
        """
        if "euclidean" in similarity_methods:
            self.method = "euclidean"
            self.distance = similarityCalculator.calc_euclidean_dist(query_hist, db_hist)
        
        if "L1_dist" in similarity_methods:
            self.method = "L1_dist"
            self.distance  = similarityCalculator.calc_L1_dist(query_hist, db_hist)
        
        if "x2_dist" in similarity_methods:
            self.method = "x2_dist"
            self.distance  = similarityCalculator.calc_x2_dist(query_hist, db_hist)
        
        if "intersection" in similarity_methods:
            self.method = "intersection"
            self.distance = similarityCalculator.calc_hist_intesection(query_hist, db_hist)
        
        if "hellinger" in similarity_methods:
            self.method = "hellinger"
            self.distance = similarityCalculator.calc_hellinger_kernel(query_hist, db_hist)
            
        if "correlation" in similarity_methods:
            self.method = "correlation"
            self.distance = similarityCalculator.calc_correlation(query_hist, db_hist)