from painting import Painting
from cv2 import cv2

def calc_equalize_hist(self, painting):
    """
    Do an image processing of contrast adjustment using the image's histogram
    Args:
        - painting: a painting instance object
    Returns
        An equalized histogram of that painting
    """

    if painting.color_space not in ['BGR', 'GRAY']:
        raise ValueError('Error: the implicit image must be on BGR color space')
    if self._color_space != 'GRAY':
        im = cv2.cvtColor(painting.img, cv2.COLOR_BGR2GRAY)
    else:
        im = painting.img

    return cv2.equalizeHist(im)

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

def calc_histogram(painting, channel, hist_size=[256], ranges=[0, 256], mask=None):
    """
    Args:
        - painting: a painting instance object
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
    
    return cv2.calcHist([painting.img], [channel], mask, hist_size, ranges)