from image import Image
import cv2

class Histogram(object):
    """
    Class to define the histogram object
    """
   
    @staticmethod
    def calc_histogram(image, channel, hist_size=[256], ranges=[0,256], mask=None):
        """
        Args:
            - image: an instance of image class
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

        return cv2.calcHist([image.img], [channel], mask, hist_size, ranges)
    
    @staticmethod
    def calc_equalize_hist(image):
        """
        The image must be BGR color space.
        Do an image processing of contrast adjustment using the image's histogram
        Args:
            - image: an instance of image class
        """

        im = cv2.cvtColor(image.img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(im)

    
