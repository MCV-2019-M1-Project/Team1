from sklearn.cluster import KMeans
import itertools
import numpy as np
import os
try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2
from texture_descriptors import LBP


def classify_images_in_clusters(n_clusters, n_best_results_to_show, forced_class = False):
    #Import bbdd paintings

    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    bbdd_paintings = []
    for filename in sorted(os.listdir(path)):
        painting = cv2.imread(os.path.join(path, filename))
        bbdd_paintings.append(painting)
    if not forced_class:
        #to run with kmeans with different descriptors
        
        #predefined centroid init values for kmeans
        centroid_imgs = [13, 9, 22, 268, 241, 235, 187, 170, 108, 80]
    
        #Compute descriptors
        descriptors = texture_histogram_descriptors(bbdd_paintings)
        
        #get centroid descriptors for defined centroids 
        calc_centroids = get_centroids(centroid_imgs, descriptors)
        #k_means clustering
        km = KMeans(n_clusters, calc_centroids).fit(descriptors)
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
    
    elif forced_class:
        #to run without kmeans hue classification
        images_in_clusters = forced_classifier(bbdd_paintings)
        
    #Show results
    for i in range(len(images_in_clusters)):

        if(len(images_in_clusters[i])>=4):
            numpy_horizontal_concat1 = np.concatenate((cv2.resize(images_in_clusters[i][0][0],(250,250)), cv2.resize(images_in_clusters[i][1][0],(250,250))), axis=1)
            numpy_horizontal_concat2 = np.concatenate((cv2.resize(images_in_clusters[i][2][0],(250,250)), cv2.resize(images_in_clusters[i][3][0],(250,250))), axis=1)
            numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
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
        painting = cv2.cvtColor(painting,cv2.COLOR_BGR2HSV)
        if color_channel==0 or color_channel==1 or color_channel==2:
            col_hist = cv2.calcHist(painting,[color_channel], None, [20], [0,256])
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
#        paintinghsv = cv2.cvtColor(painting,cv2.COLOR_BGR2HSV)
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

def get_centroids(cent_imgs, X):
#    cent_imgs = [13, 9, 22, 268, 241, 235, 187, 170, 108, 80]
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
    hue_variance_descriptor = hue_variance_descriptor

    saturation_mean_descriptor = saturation_mean_descriptor
    saturation_variance_descriptor = saturation_variance_descriptor

    value_mean_descriptor = [v * 2.0 for v in value_mean_descriptor]
    value_variance_descriptor = value_variance_descriptor

    laplacian_mean_descriptor = [v * 3.0 for v in laplacian_mean_descriptor]
    laplacian_variance_descriptor = laplacian_variance_descriptor

    sobel_phase_mean_descriptor = [v * 4.0 for v in sobel_phase_mean_descriptor]
    sobel_phase_variance_descriptor = sobel_phase_variance_descriptor

    #Join descriptors and return
    descriptors = []
    for i in range(len(hue_mean_descriptor)):
        descriptor = []
        descriptor.append(hue_mean_descriptor[i])
        descriptor.append(hue_variance_descriptor[i])

        descriptor.append(value_mean_descriptor[i])
        #descriptor.append(value_variance_descriptor[i])

        descriptor.append(laplacian_mean_descriptor[i])
        #descriptor.append(laplacian_variance_descriptor[i])

        descriptor.append(sobel_phase_mean_descriptor[i])
        #descriptor.append(sobel_phase_variance_descriptor[i])

        descriptors.append(descriptor)
    return descriptors

def forced_classifier(paintings):

    clust1=[]
    clust2=[]
    clust3=[]
    clust4=[]
    clust5=[]
    clust6=[]
    clust7=[]
    clust8=[]
    clust9=[]
    clust10=[]
    clusters=[]
    for painting in paintings:
            paintinghsv = cv2.cvtColor(painting,cv2.COLOR_BGR2HSV)
            H_hist = cv2.calcHist(paintinghsv,[0], None, [256], [0,256])
            S_hist = cv2.calcHist(paintinghsv,[1], None, [100], [0,100])
            V_hist = cv2.calcHist(paintinghsv,[2], None, [256], [0,256])
            max_color = (H_hist == max(H_hist)).flatten()
            ocurrences = max(H_hist)
            
            most_probable_color = np.where(max_color==True)[0]
#            paint_col = np.empty((0))
            if most_probable_color[0]<=4:
                paint_col = [painting, ocurrences]
                clust1.append(paint_col)
            elif 4<most_probable_color[0]<=7:
                paint_col = [painting, ocurrences]
                clust2.append(paint_col)
            elif 7<most_probable_color[0]<=11:
                paint_col = [painting, ocurrences]
                clust3.append(paint_col)
            elif 11<most_probable_color[0]<=12:
                paint_col = [painting, ocurrences]
                clust4.append(paint_col)
            elif 12<most_probable_color[0]<=15:
                paint_col = [painting, ocurrences]                
                clust5.append(paint_col)
            elif 15<most_probable_color[0]<=18:
                paint_col = [painting, ocurrences]
                clust6.append(paint_col)
            elif 18<most_probable_color[0]<=20:
                paint_col = [painting, ocurrences]
                clust7.append(paint_col)
            elif 20<most_probable_color[0]<=25:
                paint_col = [painting, ocurrences]
                clust8.append(paint_col)
            elif 25<most_probable_color[0]<=230:
                paint_col = [painting, ocurrences]
                clust9.append(paint_col)
            elif 230<most_probable_color[0]<=256:
                paint_col = [painting, ocurrences]
                clust10.append(paint_col)
    for i in range (1,11):
        clust = locals()["clust"+str(i)]
        clust.sort(key=lambda x: x[1], reverse=True)
        def_clust = []
        if len(clust)>=5:
            [def_clust.append(clust[i]) for i in range(0,5)]
            clusters.append(def_clust)
        else:
            clusters.append(def_clust)
    return clusters

def normalize(list, new_min, new_max):
    minim = min(list)
    range = float(max(list) - minim)
    new_range = float(new_max - new_min)
    ret = [float(i - minim) / range * new_range + new_min for i in list]
    return ret

#Example of usage
classify_images_in_clusters(10, 5, forced_class=False)
