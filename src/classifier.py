from sklearn.cluster import KMeans
import itertools
import numpy as np
import os
from cv2 import cv2

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


def classify_images_in_clusters(n_clusters, n_best_results_to_show):
    #Import bbdd paintings
    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    bbdd_paintings = []
    for filename in sorted(os.listdir(path)):
        painting = cv2.imread(os.path.join(path, filename))
        bbdd_paintings.append(painting)

    #Compute descriptors
    descriptors = hsv_mean_variance_gradient_mean_angle_descriptors(bbdd_paintings)

    #k_means clustering
    km = KMeans(n_clusters, 'k-means++').fit(descriptors)
    distances = km.transform(descriptors)
    labels = km.labels_

    #Sort results and obtain n_best_results_to_show
    #Sort by cluster
    images_in_clusters = [ [] for i in range(n_clusters) ]
    for i in range(distances.shape[0]):
        image = bbdd_paintings[i]
        cluster = labels[i]
        distance = min(distances[i])
        images_in_clusters[cluster].append([image, distance])
    #Sort by distance to cluster center
    for images_in_cluster in images_in_clusters:
        images_in_cluster.sort(key=lambda x: x[1])
    #Obtain n_best_results_to_show
    images_in_clusters = [images_in_cluster[0:n_best_results_to_show] for images_in_cluster in images_in_clusters]

    #Show results
    for i in range(len(images_in_clusters)):
        if(len(images_in_clusters[i])>=4):
            numpy_horizontal_concat1 = np.concatenate((cv2.resize(images_in_clusters[i][0][0],(500,500)), cv2.resize(images_in_clusters[i][1][0],(500,500))), axis=1)
            numpy_horizontal_concat2 = np.concatenate((cv2.resize(images_in_clusters[i][2][0],(500,500)), cv2.resize(images_in_clusters[i][3][0],(500,500))), axis=1)
            numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
            cv2.imshow('cluster_'+str(i),numpy_vertical_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb_gradientxy_mean_and_variance_descriptors(bbdd_paintings):
    descriptors = []
    for painting in bbdd_paintings:
        descriptor = []

        #COLOR MEAN
        #Red mean color descriptor
        red_channel = painting[:,:,0]
        red_average = red_channel.mean(axis=0).mean(axis=0)
        descriptor.append(red_average)
        #Green mean color descriptor
        green_channel = painting[:,:,1]
        green_average = green_channel.mean(axis=0).mean(axis=0)
        descriptor.append(green_average)
        #Blue mean color descriptor
        blue_channel = painting[:,:,0]
        blue_average = blue_channel.mean(axis=0).mean(axis=0)
        descriptor.append(blue_average)

        #COLOR VARIANCE
        #Red variance color descriptor
        red_variance = np.var(red_channel, dtype=np.float64)
        descriptor.append(red_variance)
        #Green variance color descriptor
        green_variance = np.var(green_channel, dtype=np.float64)
        descriptor.append(green_variance)
        #Blue variance color descriptor
        blue_variance = np.var(blue_channel, dtype=np.float64)
        descriptor.append(blue_variance)

        #GRADIENT MEAN
        image_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)
        #X gradient mean descriptor
        x_grad_average = sobelx.mean(axis=0).mean(axis=0)
        descriptor.append(x_grad_average)
        #Y gradient mean descriptor
        y_grad_average = sobely.mean(axis=0).mean(axis=0)
        descriptor.append(y_grad_average)

        #GRADIENT VARIANCE
        #X gradient variance descriptor
        x_grad_variance = np.var(sobelx, dtype=np.float64)
        descriptor.append(x_grad_variance)
        #Y gradient variance descriptor
        y_grad_variance = np.var(sobely, dtype=np.float64)
        descriptor.append(y_grad_variance)

        #Append descriptor
        descriptors.append(descriptor)
    return descriptors

def hsv_mean_variance_gradient_mean_angle_descriptors(bbdd_paintings):
    #Initialize descriptors
    hue_mean_descriptor = []
    saturation_mean_descriptor = []
    value_mean_descriptor = []

    hue_variance_descriptor = []
    saturation_variance_descriptor = []
    value_variance_descriptor = []

    laplacian_mean_descriptor = []
    laplacian_variance_descriptor = []

    sobel_phase_mean_descriptor = []
    sobel_phase_variance_descriptor = []

    #Compute descriptors
    for painting in bbdd_paintings:
        image_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
        image_hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
        sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)
        sobel_phase = cv2.phase(sobelx,sobely,angleInDegrees=True)
        laplacian = cv2.Laplacian(image_gray,cv2.CV_64F)
        hue = image_hsv[:,:,0]
        saturation = image_hsv[:,:,1]
        value = image_hsv[:,:,2]

        #HUE
        hue_average = hue.mean(axis=0).mean(axis=0)
        hue_mean_descriptor.append(hue_average)
        hue_variance = np.var(hue, dtype=np.float64)
        hue_variance_descriptor.append(hue_variance)

        #SATURATION
        saturation_average = saturation.mean(axis=0).mean(axis=0)
        saturation_mean_descriptor.append(saturation_average)
        saturation_variance = np.var(saturation, dtype=np.float64)
        saturation_variance_descriptor.append(saturation_variance)

        #VALUE
        value_average = value.mean(axis=0).mean(axis=0)
        value_mean_descriptor.append(value_average)
        value_variance = np.var(value, dtype=np.float64)
        value_variance_descriptor.append(value_variance)

        #GRADIENT VALUE
        laplacian_average = laplacian.mean(axis=0).mean(axis=0)
        laplacian_mean_descriptor.append(laplacian_average)
        laplacian_variance = np.var(laplacian, dtype=np.float64)
        laplacian_variance_descriptor.append(laplacian_variance)

        #GRADIENT VALUE
        sobel_phase_average = sobel_phase.mean(axis=0).mean(axis=0)
        sobel_phase_mean_descriptor.append(sobel_phase_average)
        sobel_phase_variance = np.var(sobel_phase, dtype=np.float64)
        sobel_phase_variance_descriptor.append(sobel_phase_variance)

    #Normalize descriptors
    hue_mean_descriptor = normalize(hue_mean_descriptor, 0.0, 1.0)
    hue_variance_descriptor = normalize(hue_variance_descriptor, 0.0, 1.0)

    saturation_mean_descriptor = normalize(saturation_mean_descriptor, 0.0, 1.0)
    saturation_variance_descriptor = normalize(saturation_variance_descriptor, 0.0, 1.0)

    value_mean_descriptor = normalize(value_mean_descriptor, 0.0, 1.0)
    value_variance_descriptor = normalize(value_variance_descriptor, 0.0, 1.0)

    laplacian_mean_descriptor = normalize(laplacian_mean_descriptor, 0.0, 1.0)
    laplacian_variance_descriptor = normalize(laplacian_variance_descriptor, 0.0, 1.0)

    sobel_phase_mean_descriptor = normalize(sobel_phase_mean_descriptor, 0.0, 1.0)
    sobel_phase_variance_descriptor = normalize(sobel_phase_variance_descriptor, 0.0, 1.0)

    #Join descriptors and return
    descriptors = []
    for i in range(len(hue_mean_descriptor)):
        descriptor = []
        descriptor.append(hue_mean_descriptor[i])
        descriptor.append(hue_variance_descriptor[i])

        descriptor.append(laplacian_mean_descriptor[i])
        descriptor.append(laplacian_variance_descriptor[i])

        descriptor.append(sobel_phase_mean_descriptor[i])
        descriptor.append(sobel_phase_variance_descriptor[i])

        descriptors.append(descriptor)
    return descriptors

def normalize(list, new_min, new_max):
    minim = min(list)
    range = float(max(list) - minim)
    new_range = float(new_max - new_min)
    ret = [float(i - minim) / range * new_range + new_min for i in list]
    return ret

#Example of usage
classify_images_in_clusters(10,4)
