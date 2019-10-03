from similarityCalculator import similarityCalculator

class Distance:
    
    def __init__(self, query_img, db_img, db_img_filename):
        self.query_img = query_img
        self.db_img = db_img
        self.db_img_filename = db_img_filename
    
    def method(self):
        
        return self.method
    
    def distance(self):
        
        return self.distance
    
    def  db_img_filename(self):
        
        return self.db_img_filename
    
    def calc_dist(self, query_hist, db_hist, num_similarities, similarity_method):
        """
        Calc the desired distance
        
        """
        if similarity_method == "euclidean":
            self.method = "euclidean"
            self.distance = similarityCalculator.calc_euclidean_dist(query_hist, db_hist)
        
        elif similarity_method == "L1_dist" :
            self.method = "L1_dist"
            self.distance  = similarityCalculator.calc_L1_dist(query_hist, db_hist)
        
        elif similarity_method == "x2_dist":
            self.method = "x2_dist"
            self.distance  = similarityCalculator.calc_x2_dist(query_hist, db_hist)
        
        elif similarity_method == "intersection":
            self.method = "intersection"
            self.distance = similarityCalculator.calc_hist_intesection(query_hist, db_hist)
        
        elif similarity_method == "hellinger":
            self.method = "hellinger"
            self.distance = similarityCalculator.calc_hellinger_kernel(query_hist, db_hist)
            
        elif similarity_method == "correlation":
            self.method = "correlation"
            self.distance = similarityCalculator.calc_correlation(query_hist, db_hist)