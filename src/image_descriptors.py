from painting import Painting
from cv2 import cv2
import numpy as np


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
        - mask: None for full image or list of masks
    Returns
        The normalized historgram of the image a 256x1 array by default
    """
    if masks is None:
        return cv2.normalize(cv2.calcHist([painting.img], channel, None, hist_size, ranges), 0.0, 1.0)
    else:
        hists = np.empty(shape=[0])
        for i in masks:
            hist_aux = cv2.normalize(cv2.calcHist([painting.img], channel, i, hist_size, ranges), 0.0, 1.0)
            hists = np.append(hists, hist_aux)
        return hists

def calc_histogram_1D_luminance_and_2D_chrominance(painting, masks=None):
    """
    Calculates the normalized 1D histogram of the iluminance and concatenates it with the normalized 2D histogram of the chrominance
    Args:
        - painting: a painting instance object
        - mask: None for full image or list of masks
    Returns
        The normalized 1D histogram of the iluminance concatenated with the normalized 2D histogram of the chrominance for every mask
    """
    ##Convert to hsv color space
    painting.color_space = 'YCrCb'
    if masks is None:
        #Compute histogram 2D
        hist2D =calc_histogram(painting, [1,2], [256, 256], [0, 256, 0, 256], None)
        #Normalize histogram 2D
        hist2D_norm = cv2.normalize(hist2D, 0.0, 1.0)
        #Flatten 2D histogram
        hist2D_flatten = hist2D_norm.flatten()
        #Compute histogram 1D
        hist1D =calc_histogram(painting, [0], [256], [0, 256], None)
        #Normalize histogram 1D
        hist1D_norm = cv2.normalize(hist1D, 0.0, 1.0)
        #Join and return features
        return np.append(hist2D_flatten, hist1D_norm)
    else:
        hists = np.empty(shape=[0])
        for i in masks:
            #Compute histogram 2D
            hist2D =calc_histogram(painting, [1,2], [256, 256], [0, 256, 0, 256], [i])
            #Normalize histogram 2D
            hist2D_norm = cv2.normalize(hist2D, 0.0, 1.0)
            #Flatten 2D histogram
            hist2D_flatten = hist2D_norm.flatten()
            #Compute histogram 1D
            hist1D =calc_histogram(painting, [0], [256], [0, 256], [i])
            #Normalize histogram 1D
            hist1D_norm = cv2.normalize(hist1D, 0.0, 1.0)
            #Join and append features to hists variable
            hists = np.append(hists, hist1D_norm)
            hists = np.append(hists, hist2D_flatten)
        return hists

def calc_block_masks(painting, n_blocks):
    """
    Computes the masks for the image size given the number of divisions to make horizontally and vertically
    Args:
        - painting: a painting instance object
        - n_blocks: number of horizontal and vertical blocks the image will be divided in
    Returns
        A list with n_blocks * n_blocks items where every item is a mask of the image
    """
    masks = []
    mask_original = np.zeros((painting.img.shape[0], painting.img.shape[1]), np.uint8)
    for i in range(n_blocks):
        for j in range(n_blocks):
            mask_aux = mask_original
            mask_aux[int(i*painting.img.shape[0]/n_blocks):int((i + 1)*painting.img.shape[0]/n_blocks)][int(j*painting.img.shape[0]/n_blocks):int((j + 1)*painting.img.shape[0]/n_blocks)] = 1
            masks.append(mask_aux)
    return masks
