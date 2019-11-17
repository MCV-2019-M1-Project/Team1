from sklearn.cluster import KMeans
import itertools
import numpy as np
import os
import cv2
from texture_descriptors import LBP

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
    descriptors = paper_descriptors(bbdd_paintings)
    print("All descriptors found")
    #calc_centroids = get_centroids(descriptors)
    #k_means clustering
    km = KMeans(n_clusters).fit(descriptors)
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
    half_ones = np.zeros((250,125,3), np.uint8)
    for i in range(len(images_in_clusters)):
        print(len(images_in_clusters[i]))
        if(len(images_in_clusters[i])>4):
            numpy_horizontal_concat1 = np.concatenate((cv2.resize(images_in_clusters[i][0][0],(250,250)), cv2.resize(images_in_clusters[i][1][0],(250,250))), axis=1)
            numpy_horizontal_concat2 = np.concatenate((cv2.resize(images_in_clusters[i][2][0],(250,250)), cv2.resize(images_in_clusters[i][3][0],(250,250))), axis=1)
            numpy_horizontal_concat3 = np.concatenate((half_ones, cv2.resize(images_in_clusters[i][4][0],(250,250)), half_ones), axis=1)
            numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2, numpy_horizontal_concat3), axis=0)
            cv2.imshow('cluster_'+str(i),numpy_vertical_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_histogram_descriptors(bbdd_paintings, color_channel=0):
    """
    computes color histogram for the specified channel color_channel=(0, 1, 2)
    or for all color channels and concatenates the histograms color_channel=(-1)
    """
    X=np.empty((0))
    for painting in bbdd_paintings:
        if color_channel==0 or color_channel==1 or color_channel==2:
            col_hist = cv2.calcHist(painting,[color_channel], None, [256], [0,256])
        elif color_channel==-1:
            col1_hist = cv2.calcHist(painting,[0], None, [256], [0,256])
            col2_hist = cv2.calcHist(painting,[1], None, [256], [0,256])
            col3_hist = cv2.calcHist(painting,[2], None, [256], [0,256])
            col_hist = np.concatenate([col1_hist, col2_hist, col3_hist])

        if X.size>0:
            X = np.vstack((X, col_hist.T))
        else:
            X = col_hist.T
    return X

def texture_histogram_descriptors(bbdd_paintings):
    X=np.empty((0))
    for painting in bbdd_paintings:
        image_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)

        text_hist = np.asarray(LBP(image_gray))

        if X.size>0:
            X = np.vstack((X, text_hist.T))
        else:
            X = text_hist.T
    return X

def texturecolor_histogram_descriptors(bbdd_paintings, color_channel=0):
    X=np.empty((0))
    for painting in bbdd_paintings:

        if color_channel==0 or color_channel==1 or color_channel==2:
            col_hist = cv2.calcHist(painting,[color_channel], None, [256], [0,256])
        elif color_channel==-1:
            col1_hist = cv2.calcHist(painting,[0], None, [256], [0,256])
            col2_hist = cv2.calcHist(painting,[1], None, [256], [0,256])
            col3_hist = cv2.calcHist(painting,[2], None, [256], [0,256])
            col_hist = np.concatenate([col1_hist, col2_hist, col3_hist])

        image_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)

        text_hist = np.asarray(LBP(image_gray))

        col_text_hist = np.concatenate([col_hist, text_hist])

        if X.size>0:
            X = np.vstack((X, col_text_hist.T))
        else:
            X = col_text_hist.T
    return X

def get_centroids(X):
    cent_imgs = [13, 9, 22, 268, 241, 235, 187, 168, 108, 80]
    centroid = np.empty((0))
    for cent in cent_imgs:
        if centroid.size>0:
                centroid = np.vstack((centroid, X[cent][:]))
        else:
                centroid = X[cent][:]
    return centroid

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

    #Weight descriptors
    hue_mean_descriptor = [v * 4.0 for v in hue_mean_descriptor]
    hue_variance_descriptor = [v * 2.0 for v in hue_variance_descriptor]

    saturation_mean_descriptor = saturation_mean_descriptor
    saturation_variance_descriptor = saturation_variance_descriptor

    value_mean_descriptor = [v * 1.0 for v in value_mean_descriptor]
    value_variance_descriptor = [v * 3.0 for v in value_variance_descriptor]

    laplacian_mean_descriptor = [v * 1.0 for v in laplacian_mean_descriptor]
    laplacian_variance_descriptor = [v * 3.0 for v in laplacian_variance_descriptor]

    sobel_phase_mean_descriptor = [v * 2.0 for v in sobel_phase_mean_descriptor]
    sobel_phase_variance_descriptor = [v * 3.0 for v in sobel_phase_variance_descriptor]

    #Join descriptors and return
    descriptors = []
    for i in range(len(hue_mean_descriptor)):
        descriptor = []
        descriptor.append(hue_mean_descriptor[i])
        descriptor.append(hue_variance_descriptor[i])

        #descriptor.append(value_mean_descriptor[i])
        descriptor.append(value_variance_descriptor[i])

        descriptor.append(laplacian_mean_descriptor[i])
        descriptor.append(laplacian_variance_descriptor[i])

        descriptor.append(sobel_phase_mean_descriptor[i])
        #descriptor.append(sobel_phase_variance_descriptor[i])

        descriptors.append(descriptor)
    return descriptors

def paper_descriptors(bbdd_paintings):
    #Initialize descriptors
    hue_mean_descriptor = []
    saturation_mean_descriptor = []

    hue_count_descriptors = []

    composition_descriptors = []

    lbp_descriptors = []

    #Compute descriptors
    for painting_full_size in bbdd_paintings:
        painting = cv2.resize(painting_full_size, (500,500))
        image_gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
        image_hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
        sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)
        sobel_phase = cv2.phase(sobelx,sobely,angleInDegrees=True)
        laplacian = cv2.Laplacian(image_gray,cv2.CV_64F)
        hue = image_hsv[:,:,0]
        saturation = image_hsv[:,:,1]
        value = image_hsv[:,:,2]

        #COLOR FEATURES
        #HUE
        hue_average = hue.mean(axis=0).mean(axis=0)
        hue_mean_descriptor.append(hue_average)

        #SATURATION
        saturation_average = saturation.mean(axis=0).mean(axis=0)
        saturation_mean_descriptor.append(saturation_average)

        #HUE COUNT
        hue_count_descriptor_aux = []
        for i in range(20):
            hue_histogram = cv2.calcHist([image_hsv], [0], None, [9], [i*9, (i+1)*9])
            hue_max = max(hue_histogram)
            hue_count = sum(hue_histogram>hue_max*0.1)
            hue_count_descriptor_aux.append(hue_count[0]/180.0)
        hue_count_descriptors.append(hue_count_descriptor_aux)
        '''
        #COMPOSITION FEATURES
        #Image segmentation
        image_as_list = image_hsv.reshape((image_hsv.shape[0]*image_hsv.shape[1], 3))
        clt = KMeans(n_clusters = 5, init='k-means++')
        clt.fit(image_as_list)
        image_segmented_k_means = np.uint8(clt.labels_.reshape((image_hsv.shape[0], image_hsv.shape[1])))

        #Initialize descriptors
        minus_inf = 0.000000001
        areas = [0.0, 0.0, 0.0, 0.0, 0.0]
        x_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
        y_mean = [0.0, 0.0, 0.0, 0.0, 0.0]
        xs = [[], [], [], [], []]
        ys = [[], [], [], [], []]
        h = [0.0, 0.0, 0.0, 0.0, 0.0]
        s = [0.0, 0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0, 0.0]

        #Fill in descriptors
        for i in range(len(image_segmented_k_means)):
            for j in range(len(image_segmented_k_means[i])):
                label = image_segmented_k_means[i][j]
                areas[label] += 1.0
                x_mean[label] += i
                y_mean[label] += j
                xs[label].append(i)
                ys[label].append(j)
                h[label] += image_hsv[i,j,0]
                s[label] += image_hsv[i,j,1]
                v[label] += image_hsv[i,j,2]

        for i in range(len(areas)):
            if(areas[i]==0.0):
                print("DIV BY 0")
                x_mean[i] = 250.0
                y_mean[i] = 250.0
                h[i] = 0.5
                s[i] = 0.5
                v[i] = 0.5
            else:
                x_mean[i] = float(x_mean[i]) / float(areas[i])
                y_mean[i] = float(y_mean[i]) / float(areas[i])
                h[i] = float(h[i]) / (float(areas[i]) * 180.0)
                s[i] = float(s[i]) / (float(areas[i]) * 256.0)
                v[i] = float(v[i]) / (float(areas[i]) * 256.0)

        difxy2 = [0.0, 0.0, 0.0, 0.0, 0.0]
        difxy3 = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i in range(len(xs)):
            for j in range(len(xs[i])):
                difxy2[i] += (xs[i][j] - x_mean[i])**2 + (ys[i][j] - y_mean[i])**2
                difxy3[i] += (xs[i][j] - x_mean[i])**3 + (ys[i][j] - y_mean[i])**3

        for i in range(len(difxy2)):
             if(areas[i]==0.0):
                 difxy2[i] = 0.5
                 difxy3[i] = 0.5
                 x_mean[i] = 0.5
                 y_mean[i] = 0.5
             else:
                 difxy2[i] = float(difxy2[i]) / (float(areas[i]) * 500.0**2)
                 difxy3[i] = float(difxy3[i]) / (float(areas[i]) * 500.0**3)
                 x_mean[i] = float(x_mean[i]) / 500.0
                 y_mean[i] = float(y_mean[i]) / 500.0

        #Save descriptors
        descriptor = []
        indexs_max_area = sorted(range(len(areas)), reverse=True, key=areas.__getitem__)
        indexs_max_area = indexs_max_area[0:3]
        for i in indexs_max_area:
            descriptor.append(x_mean[i])
            descriptor.append(y_mean[i])
            descriptor.append(difxy2[i])
            descriptor.append(difxy3[i])
        for i in range(5):
            descriptor.append(h[i])
            descriptor.append(s[i])
            descriptor.append(v[i])
        composition_descriptors.append(descriptor)
        '''
        #LBP histogram
        lbp_histo = np.asarray(LBP(image_gray.astype('uint8'), resize_level=1, histogram_size=[8]))
        print(lbp_histo.flatten().shape)
        lbp_descriptors.append(lbp_histo.flatten())

    #Normalize descriptors
    hue_mean_descriptor = normalize(hue_mean_descriptor, 0.0, 1.0)

    saturation_mean_descriptor = normalize(saturation_mean_descriptor, 0.0, 1.0)

    #Join descriptors and return
    descriptors = []
    for i in range(len(hue_mean_descriptor)):
        descriptor = []
        descriptor.append(hue_mean_descriptor[i])

        descriptor.append(saturation_mean_descriptor[i])

        for j in range(len(hue_count_descriptors[i])):
            descriptor.append(hue_count_descriptors[i][j])

        #for j in range(len(composition_descriptors[i])):
        #    descriptor.append(composition_descriptors[i][j])

        for j in range(len(lbp_descriptors[i])):
            descriptor.append(lbp_descriptors[i][j])

        descriptors.append(descriptor)
    return descriptors

def normalize(list, new_min, new_max):
    minim = min(list)
    range = float(max(list) - minim)
    new_range = float(new_max - new_min)
    ret = [float(i - minim) / range * new_range + new_min for i in list]
    return ret

#Example of usage
classify_images_in_clusters(11,5)
