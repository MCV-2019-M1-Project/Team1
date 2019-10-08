from painting import Painting


class MuseumItem(object):
    def __init__(self, painting=None, histogram=None):
        """
        Creates a new Museum Item with or not an painting and histogram
        Args:
            - painting: a painting object instnace
            - histogram: the histogram associated to that painting
        """

        self._painting = painting
        self._histogram = histogram

    @property
    def painting(self):
        """
        Returns:
            the stored painting instance
        """

        return self._painting

    @property
    def histogram(self):
        """
        Returns:
            the histogram associated to that image
        """

        return self._histogram

    @image.setter
    def image(self, painting):
        """
        Set a new painting
        Args:
            - image: a new painting object instance
        """

        self._painting = painting

    @histogram.setter
    def histogram(self, histogram):
        """
        Set histogram
        Args:
            - histogram: set a new histogram
        """

        self._histogram = histogram