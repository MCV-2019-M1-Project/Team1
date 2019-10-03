from image import Image

class MuseumItem(object):
    def __init__(self, image=None, histogram=None):
        """
        Creates a new Museum Item with or not an image and histogram
        Args:
            - image: a image object instnace
            - histogram: the histogram associated to that image
        """

        self.image = image
        self.histogram = histogram

    @property
    def image(self):
        """
        Returns:
            the stored image instance
        """

        return self._image
    
    @property
    def histogram(self):
        """
        Returns:
            the histogram associated to that image
        """

        return self._histogram

    @image.setter
    def image(self, image):
        """
        Set a new image
        Args:
            - image: a new image object instance
        """

        self._image = image

    @histogram.setter
    def histogram(self, histogram):
        """
        Set histogram
        Args:
            - histogram: set a new histogram
        """

        self._histogram = histogram