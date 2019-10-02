import numpy
import cv2


class Image(object):
    """
    Class to define a Image
    """
    def __init__(self, filename):
        """
        Initialize an instance of Image
        """

        self._img = cv2.imread(filename)

    @property
    def img(self):
        return self._img
