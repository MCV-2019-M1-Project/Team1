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

def calc_normalized_histogram(image, channel, hist_size=[256], ranges=[0, 256], masks=None):
    """
    Args:
        - image: image where the histogram is going to be computed
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
    #Convert to hsv color space
    painting.color_space = 'YCrCb'
    if masks is None:
        #Compute histogram 2D
        hist2D =calc_normalized_histogram(painting.img, [1,2], [256, 256], [0, 256, 0, 256], None)
        #Flatten 2D histogram
        hist2D_flatten = hist2D.flatten()
        #Compute histogram 1D
        hist1D =calc_normalized_histogram(painting.img, [0], [256], [0, 256], None)
        #Join and return features
        return np.append(hist2D_flatten, hist1D)
    else:
        hists = np.empty(shape=[0])
        for i in masks:
            #Compute histogram 2D
            hist2D =calc_normalized_histogram(painting.img, [1,2], [256, 256], [0, 256, 0, 256], [i])
            #Flatten 2D histogram
            hist2D_flatten = hist2D.flatten()
            #Compute histogram 1D
            hist1D =calc_normalized_histogram(painting.img, [0], [256], [0, 256], [i])
            #Join and return features
            hist_joined = np.append(hist2D_flatten, hist1D)
            hists = np.append(hists, hist_joined)
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

def compute_multires_histograms(painting, list_gauss_filter_sizes, channel, hist_size=[256], ranges=[0, 256], masks=None):
    """
    Computes the multiresolution histograms where each resolution is obtained applying a gaussian filter of the size list_gauss_filter_sizes[i]*list_gauss_filter_sizes[i]
    Args:
        - painting: a painting instance object
        - list_gauss_filter_sizes: list of the sizes of the gaussaian filter that are going to be used to obtain the different resolutions
        - channel:
            * 0 if is a grayscale image
            * 0 calculate blue histogram
            * 1 calculate green histogram
            * 2 calculate red histogram
        - hist_size: this represents our BIN count. By default is full scale.
        - ranges: histogram range. By default is full range
        - mask: None for full image or list of masks
    Returns
        Array of the concatenation of the histograms at the different resolutions
    """
    hists = np.empty(shape=[0])
    if masks is None:
        for i in list_gauss_filter_sizes:
            im_resized = cv2.GaussianBlur(painting.img,(i,i),0)
            hist_aux = calc_normalized_histogram(im_resized, channel, hist_size, ranges, masks)
            hists = np.append(hists, hist_aux)
        return hists
    else:
        for j in masks:
            for i in list_gauss_filter_sizes:
                im_resized = cv2.GaussianBlur(painting.img,(i,i),0)
                hist_aux = calc_normalized_histogram(im_resized, channel, hist_size, ranges, [j])
                hists = np.append(hists, hist_aux)
        return hists
