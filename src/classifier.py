from sklearn.cluster import KMeans
import itertools
import numpy as np

def k_means_classifier(X, n_clust=10, centroids='random'):
    """
    compute k means and return the 5 best labeled paintings for each cluster
    n_clust: number of clusters in which classify the paintings
    centroids: init centroids or method to find init centroids ('k-means++', 'random')
    X: array where every row is one example (image) and every column is a feature.
    kmeans.labels_ = Labels of each point
    kmeans.cluster_centers_ = cluster centers
    
    Returns a List of a list with a tuple which contains the distances to centroids and the image number
    """
    km = KMeans(n_clusters=n_clust, init=centroids)
    compute_km = km.fit(X)
    dist = km.transform(X)
    L = compute_km.labels_
    C = km.cluster_centers_
    img_indx = list(range(0,X.shape[0]))
    
    dist_cluster = []
    for row in dist:
        dist_cluster.append(min(row))
        
    dist_clust_list = []
    for i in range(0, n_clust):
        lab = L==i
        distances_by_clust = []
        for p in range(0, X.shape[0]):
            if lab[p]==True:
                distances_by_clust.append(dist_cluster[p])
            else:
                distances_by_clust.append('False')
#        distances_by_clust = list(itertools.compress(dist_cluster, lab))
        clusters_labeled = list(zip(distances_by_clust,img_indx))
        dist_clust_list.append(clusters_labeled)
        
    list_best_labels = []
    for elem in dist_clust_list:
        elem.sort(key=lambda x: str(x[0]))
        for d in range(0, X.shape[0]):
            if elem[d][0]!= 'False':
                best_labels = elem[d:d+6]
                break
        list_best_labels.append(best_labels)
#        

    return list_best_labels

