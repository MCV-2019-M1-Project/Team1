import numpy as np
from scipy.spatial import distance
from cv2 import cv2


def euclidean(histogram1, histogram2):
    return distance.euclidean(histogram1, histogram2)


def l1_dist(histogram1, histogram2):
    return distance.cityblock(histogram1, histogram2)


def x2_dist(histogram1, histogram2):
    return np.sum(
        (histogram1 - histogram2)**2 / (histogram1 + histogram2 + 1e-6))


def intersection(histogram1, histogram2):
    return cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_INTERSECT)


def hellinger(histogram1, histogram2):
    return cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_HELLINGER)


def correlation(histogram1, histogram2):
    return cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
