from painting import Painting
from cv2 import cv2
import numpy as np

from distance import (
    euclidean,
    l1_dist,
    x2_dist,
    intersection,
    hellinger,
    correlation
)


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

def calc_normalized_histogram(image, mask, channel, hist_size=[256], ranges=[0, 256]):
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
    Returns
        The normalized historgram of the image a 256x1 array by default
    """
    return cv2.normalize(cv2.calcHist([image], channel, mask, hist_size, ranges), 0.0, 1.0)

def calc_histogram_1D_luminance_and_2D_chrominance(image, mask):
    """
    Calculates the normalized 1D histogram of the iluminance and concatenates
    it with the normalized 2D histogram of the chrominance
    Args:
        - image: image in YCrCb
    Returns
        The normalized 1D histogram of the iluminance concatenated with the normalized
        2D histogram of the chrominance
    """

    #Compute histogram 2D
    hist2D = calc_normalized_histogram(image, mask, [1,2], [256, 256], [0, 256, 0, 256])
    #Flatten 2D histogram
    hist2D_flatten = hist2D.flatten()
    #Compute histogram 1D
    hist1D = calc_normalized_histogram(image, mask, [0], [256], [0, 256])
    #Join and return features
    return np.append(hist2D_flatten, hist1D)

def calc_3histogram1D_concatenated(image, hist_size=[256, 256, 256], ranges=[0, 256, 0, 256, 0, 256]):
    """
    Calculates the normalized 3D histogram
    Args:
        - image: image
        - hist_size: this represents our BIN count. By default is full scale.
        - ranges: histogram range. By default is full range
    Returns
        The normalized 3D histogram
    """

    blue_hist = calc_normalized_histogram(image, [0], [256], [0, 256])
    green_hist = calc_normalized_histogram(image, [1], [256], [0, 256])
    red_hist = calc_normalized_histogram(image, [2], [256], [0, 256])
    return np.concatenate((red_hist, blue_hist, green_hist))

def calc_multires_images(image, list_gauss_filter_sizes):
    """
    Computes the image at different resolutions. Each resolution is obtained applying a gaussian filter of the size
    list_gauss_filter_sizes[i]*list_gauss_filter_sizes[i].
    Args:
        - image: an image
        - list_gauss_filter_sizes: list of the sizes of the gaussaian filter
                                   that are going to be used to obtain the
                                   different resolutions. They have to be odd
    Returns
        Array the images at different resolutions
    """
    images = []
    for gauss_filter in list_gauss_filter_sizes:
        im_resized = cv2.GaussianBlur(image, (gauss_filter, gauss_filter), 0)
        images.append(im_resized)
    return images

def calc_blocks_from_image(image, n_blocks):
    """
    Computes the image blocks of an image given the number of divisions to make horizontally and vertically
    Args:
        - image: an image
        - n_blocks: number of horizontal and vertical blocks the image will be divided in
    Returns
        A list with n_blocks * n_blocks items where every item is section of the original image
    """

    cropped_images = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            cropped_image = image[int(i*image.shape[0]/n_blocks):int((i + 1)*image.shape[0]/n_blocks), int(j*image.shape[1]/n_blocks):int((j + 1)*image.shape[1]/n_blocks)]
            cropped_images.append(cropped_image)
    return cropped_images

def best_color_descriptor(image, mask):
    """
    Computes best color descriptor
    Args:
        - image: an image in YCrCb color space
        - mask: None or mask to be used to compute the histograms(needs to have the same size as image)
    Returns
        Best color descriptor

    """
    output_histograms = []

    #Calculate 2x2 blocks histograms
    blocks = calc_blocks_from_image(image, 2)
    if mask is not None:
        masks = calc_blocks_from_image(mask, 2)
        for img, sub_mask in zip(blocks, masks):
            output_histograms.append(calc_histogram_1D_luminance_and_2D_chrominance(img, sub_mask))
    else:
        for img in blocks:
            output_histograms.append(calc_histogram_1D_luminance_and_2D_chrominance(img, mask))

    #Calculate 5x5 blocks histograms
    blocks = calc_blocks_from_image(image, 5)
    if mask is not None:
        masks = calc_blocks_from_image(mask, 5)
        for img, sub_mask in zip(blocks, masks):
            output_histograms.append(calc_histogram_1D_luminance_and_2D_chrominance(img, sub_mask))
    else:
        for img in blocks:
            output_histograms.append(calc_histogram_1D_luminance_and_2D_chrominance(img, mask))

    return output_histograms

def distance_best_color_descriptor(descriptor1, descriptor2, distance_method=correlation):
    """
    Computes the distance between both descriptors
    Args:
        - descriptor1: descriptor of one image obtained from best_color_descriptor()
        - descriptor1: descriptor of one image obtained from best_color_descriptor()
        - metric: coparision metric which can be:
            - euclidean
            - l1 distance
            - x2 distance
            - intersection
            - correlation
            - hellinger
    Returns
        Distance between descriptor1 and descriptor2
    """
    distance = 0.0
    for histogram_desc1, histogram_desc2 in zip(descriptor1, descriptor2):
        if distance_method == 'euclidean':
            distance += euclidean(histogram_desc1, histogram_desc2)
        elif distance_method == 'l1_dist':
            distance += l1_dist(histogram_desc1, histogram_desc2)
        elif distance_method == 'x2_dist':
            distance += x2_dist(histogram_desc1, histogram_desc2)
        elif distance_method == 'intersection':
            distance += intersection(histogram_desc1, histogram_desc2)
        elif distance_method == 'hellinger':
            distance += hellinger(histogram_desc1, histogram_desc2)
        elif distance_method == 'correlation':
            distance += correlation(histogram_desc1, histogram_desc2)
    return distance
