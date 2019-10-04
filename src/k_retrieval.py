def k_retrieval(distances_list, k):
    
    """
    
    Returns the K more similar images of the database from a list of distance obj
    
    """
    
    maximization_required = distances_list[0].maximization_required
    # Same method to all the distances of the list
    
    distances_list.sort(key=lambda x: x.distance, reverse=maximization_required)
    k_distances=distances_list[0 : k+1]

        
    return k_distances