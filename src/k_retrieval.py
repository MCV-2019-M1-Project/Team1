def k_retrieval(distances_list, similarity_methods, k):
    
    """
    
    Returns the K more similar images of the database from a list of distance obj
    
    """
    
    method=distances_list[0].method

    if method == "euclidean" or method == "L1_dist" or method == "x2_dist" or method == "hellinger":
        """
        if the similarity is computed with one of these methods we need to choose the minimum values
        
        """
        
        #sorted_distances 
        distances_list.sort(key=lambda x: x.distance)
        k_distances=distances_list[0 : k+1]
        
    else:
        """
        
        else if the similarity is computed with other methods we need to choose the maximum value (correlation/intersection)
        
        """
        
        distances_list.sort(key=lambda x: x.distance, reverse=True)
        k_distances=distances_list[0 : k+1]
        
    return k_distances