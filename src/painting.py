from pathlib import Path
from cv2 import cv2


class Painting(object):
    """
    Class to define a Painting object
    """
    def __init__(self, image_filename):
        """
        Initialize an instance of Painting. By default the image is stored on
        BGR color space
        Args:
            - image_filename: image file path
        """

        self.set_img(image_filename)

    @property
    def img(self):
        return self._img

    def set_img(self, filename):
        """
        Change the current image
        Args:
            - filename: image file path
        """

        if not Path(filename).is_file():
            raise ValueError(
                'The filename: {} does not exists'.format(filename))

        self._color_space = 'BGR'
        self._filename = filename.split('/')[-1:][0][:-4]
        self._img = cv2.imread(filename)

    @property
    def image_name(self):
        """
        Returns the image name
        """
        return int(self._filename[5:])

    @property
    def filename(self):
        """
        Returns:
            the filename
        """

        return self._filename

    @property
    def color_space(self):
        """
        Returns:
            the color space
        """

        return self._color_space

    @color_space.setter
    def color_space(self, new_color_space):
        """
        Change the current image space color to a new one
        Args:
            - new_color_space:
                * BGR
                * HSV
                * LAB
                * GRAY
                * YCrCb
        """

        if self._color_space == 'BGR':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)
            elif new_color_space == 'GRAY':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            elif new_color_space == 'YCrCb':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2YCrCb)
            else:
                raise NotImplementedError

        elif self._color_space == 'HSV':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2BGR)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2LAB)
            else:
                raise NotImplementedError

        elif self._color_space == 'LAB':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2HSV)
            elif new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)
            else:
                raise NotImplementedError

        elif self._color_space == 'GRAY':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)
            else:
                raise NotImplementedError
        elif self._color_space == 'YCrCb':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_YCrCb2BGR)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self._color_space = new_color_space
