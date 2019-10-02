import cv2


class Image(object):
    """
    Class to define a Image
    """
    def __init__(self, filename):
        """
        Initialize an instance of Image
        """

        self._color_space = 'BGR'
        self._filename = filename
        self._img = cv2.imread(filename)

    @property
    def img(self):
        return self._img

    @property
    def filename(self):
        return self._filename
    
    @property
    def color_space(self):
        return self._color_space

    def change_color_space(self, new_color_space):
        """
        Change the current image space color to a new one
        Args:
            - new_color_space:
                * BGR
                * HSV
                * LAB
        """

        if self._color_space == 'BGR':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
            if new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)
        elif self._color_space == 'HSV':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2BGR)
            if new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2LAB)
        elif self._color_space == 'LAB':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2HSV)
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)
        self._color_space = new_color_space

    def image_filtering(self, ddepth, kernel):
        """
        Apply image filtering
        Args:
            - ddepth: desired depth of the destination image
            - kernel: convolution kernel
        """

        self._img = cv2.filter2D(self._img, ddepth, kernel)