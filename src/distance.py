import numpy as np
from scipy.spatial import distance
from cv2 import cv2

class Distance:
    def __init__(self, query_museum_item, db_museum_item):
        self._query_museum_item = query_museum_item
        self._db_museum_item = db_museum_item
        self._distance = None
        self._method = None
        self._maximization = False

    @property
    def query_museum_item(self):
        return self._query_museum_item

    @property
    def db_im(self):
        return self._db_museum_item

    @property
    def method(self):
        return self._method

    @property
    def distance(self):
        return self._distance

    @property
    def maximization(self):
        """
        True if method requires maximization, False if minimization
        """
        return self._maximization

    def calc_dist(self, similarity_method):
        """
        Calc the desired distance
        
        """
        query_hist = self._query_museum_item.histogram
        db_hist = self._db_museum_item.histogram

        if similarity_method == "euclidean":
            self._method = "euclidean"
            self._distance = distance.euclidean(query_hist, db_hist)
            self._maximization = False

        elif similarity_method == "L1_dist":
            self._method = "L1_dist"
            self._distance = distance.cityblock(query_hist, db_hist)
            self._maximization = False

        elif similarity_method == "x2_dist":
            self._method = "x2_dist"
            self._distance = np.sum(
                (query_hist - db_hist)**2 / (query_hist + db_hist + 1e-6))
            self._maximization = False

        elif similarity_method == "intersection":
            self._method = "intersection"
            self._distance = cv2.compareHist(query_hist, db_hist,
                                             cv2.HISTCMP_INTERSECT)
            self._maximization = True

        elif similarity_method == "hellinger":
            self._method = "hellinger"
            self._distance = cv2.compareHist(query_hist, db_hist,
                                             cv2.HISTCMP_HELLINGER)
            self._maximization = False

        elif similarity_method == "correlation":
            self._method = "correlation"
            self._distance = cv2.compareHist(query_hist, db_hist,
                                             cv2.HISTCMP_CORREL)
            self._maximization = True
