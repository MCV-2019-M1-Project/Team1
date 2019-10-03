from pathlib import Path
from scipy.signal import find_peaks
import numpy
import cv2


class Image(object):
    """
    Class to define a Image
    """
    def __init__(self, filename):
        """
        Initialize an instance of Image. By default the image is stored on
        BGR color space
        Args:
            - filename: image file path
        """

        self.set_img(filename)

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
        """

        if new_color_space not in ['BGR', 'HSV', 'LAB']:
            raise ValueError(
                'The change color space from {} to {} is not implemented yet'.
                format(self._color_space, new_color_space))

        if self._color_space == 'BGR':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)
        elif self._color_space == 'HSV':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2BGR)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2LAB)
        elif self._color_space == 'LAB':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2HSV)
            elif new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)
        self._color_space = new_color_space

    def image_filtering(self, ddepth, kernel):
        """
        Apply image filtering
        Args:
            - ddepth: desired depth of the destination image
            - kernel: convolution kernel
        Returns:
            A the image with filtering
        """

        return cv2.filter2D(self._img, ddepth, kernel)

    def calc_histogram(self,
                       channel,
                       hist_size=[256],
                       ranges=[0, 256],
                       mask=None):
        """
        Args:
            - channel:
                * 0 if is a grayscale image
                * 0 calculate blue histogram
                * 1 calculate green histogram
                * 2 calculate red histogram
            - hist_size: this represents our BIN count. By default is full scale.
            - ranges: histogram range. By default is full range
            - mask: mask image. By default is the full image
        Returns
            The historgram of the image a 256x1 array by default
        """

        return cv2.calcHist([self._img], [channel], mask, hist_size, ranges)

    def calc_equalize_hist(self):
        """
        Do an image processing of contrast adjustment using the image's histogram
        """

        if self._color_space != 'BGR':
            raise ValueError(
                'Error: the implicit image must be on BGR color space')
        im = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(im)

    def get_background(self, height_ratio=10, distance_between_peaks=10):
        """
        Get the background of the image if it has
        Args:
            - height_ratio:
            - distance_between_peaks:
        Returns:
            The image background
        """
        def my_find_peaks(array, height_ratio, distance_between_peaks):
            peaks, _ = find_peaks(array,
                                  height=numpy.max(array) / height_ratio,
                                  distance=distance_between_peaks)
            return peaks

        if self._color_space != 'BGR':
            raise ValueError(
                'Error: the implicit image must be on BGR color space')

        gray_image = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)

        col_var = numpy.abs(numpy.gradient(gray_image.mean(0)))

        left_cd = my_find_peaks(col_var[:len(col_var) // 2], height_ratio,
                                distance_between_peaks)
        left = left_cd[0]

        right_cd = my_find_peaks(col_var[len(col_var) // 2:], height_ratio,
                                 distance_between_peaks) + len(col_var) // 2
        right = right_cd[-1]

        row_var = numpy.abs(numpy.gradient(gray_image.mean(1)))

        top_cd = my_find_peaks(row_var[:len(row_var) // 2], height_ratio,
                               distance_between_peaks)
        top = top_cd[0]

        bottom_cd = my_find_peaks(row_var[len(row_var) // 2:], height_ratio,
                                  distance_between_peaks) + len(row_var) // 2
        bottom = bottom_cd[-1]

        mask = numpy.zeros(shape=(self._img.shape[0], self._img.shape[1]),
                           dtype=numpy.uint8)
        mask[top:bottom, left:right] = 255
        return mask
