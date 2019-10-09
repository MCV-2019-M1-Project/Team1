from pathlib import Path
import os
from painting import Painting
from museum_item import MuseumItem
from export_manager import export_museum_item
from distance import (euclidean, hellinger, intersection, l1_dist, x2_dist, correlation)
import pickle
from cv2 import cv2

def k_retrieval(list, k, reverse=False):
    """
    Args:
        - list: list of integers
        - k: how many elements retrieve
    Returns
        The K lowest values from that list
    """

    return list.sort(reverse=reverse)[0:k + 1]

def load_bbdd_paitings():
    """
    Load all the paintings stored on the bbdd folder
    Returns:
        A list of Painting object instances
    """

    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    paintings = []
    for filename in sorted(os.listdir(path)):
        painting = Painting(os.path.join(path, filename))
        paintings.append(painting)
    return paintings


def load_query_paintings(query_folder):
    """
    Loads all the query paintings stored on the query folder
    Args:
        - query_folder: specifies the query_folder name
    Returns:
        A list of Painting instnaces
    """

    path = os.path.join(os.path.dirname(__file__),
                        '../{}'.format(query_folder))
    query_paintings = []
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.jpg':
            query_painting = Painting(os.path.join(path, filename))
            query_paintings.append(query_painting)
    return query_paintings


def apply_change_of_color_space(paintings, color_space):
    """
    Apply to all paintings a color space change
    Args:
        - paintings: a list of Painting object instances
        - color_space: the color space to change could be
            * BGR
            * HSV
            * LAB
            * GRAY
            * YCrCb
    """

    for painting in paintings:
        painting.color_space = color_space


def calc_image_histogram(images,
                         channel,
                         hist_size=[256],
                         ranges=[0, 256],
                         mask=None):
    """
    Calc the histogram of the images
    Args:
        - images: a list of Image object instances
        - channel:
            * 0 if is a grayscale image
            * 0 calculate blue histogram
            * 1 calculate green histogram
            * 2 calculate red histogram
        - hist_size: this represents our BIN count. By default is full scale.
        - ranges: histogram range. By default is full range
        - mask: mask image. By default is the full image
    Returns
        Returns a list of historgrams based on the given images
    """

    histograms = []
    for image in images:
        histogram = image.calc_histogram(channel, hist_size, ranges, mask)
        histograms.append(histogram)
    return histograms


def export(images, histograms):
    """
    Exports the images and histograms into data folder as museum items
    Args:
        - images: a list of Images instances
        - histograms: a list of arrays
    """

    for image, histogram in zip(images, histograms):
        museum_item = MuseumItem(image, histogram)
        export_museum_item(museum_item)


def calc_similarties(db_museum_items, query_museum_item, method):
    """
    Calc the similarity of all museum items against the query image histogram
    Args:
        db_museum_items: a list of museum items
        query_museum_item: a query museum item
        method: the similarity method
            * euclidean
            * L1_dist
            * x2_dist
            * intersection
            * hellinger
            * correlation
    Returns:
        A list of similarities of all museum items against the query image histogram
    """

    distances = []
    for db_museum_item in db_museum_items:
        if method == 'euclidean':
            distances.append(euclidean(db_museum_item.histogram, query_museum_item.histogram))
        elif method == 'L1_dist':
            distances.append(l1_dist(db_museum_item.histogram, query_museum_item.histogram))
        elif method == 'x2_dist':
            distances.append(x2_dist(db_museum_item.histogram, query_museum_item.histogram))
        elif method == 'intersection':
            distances.append(intersection(db_museum_item.histogram, query_museum_item.histogram))
        elif method == 'hellinger':
            distances.append(hellinger(db_museum_item.histogram, query_museum_item.histogram))
        elif method == 'correlation':
            distances.append(correlation(db_museum_item.histogram, query_museum_item.histogram))
        else:
            raise NotImplementedError
    return distances


def load_gt_corresps(query_folder):
    """
    Loads the correspondances file of the query folder and returns the list of real correspondances
    """

    path = os.path.join(os.path.dirname(__file__)+'%s..%s' % (os.sep, os.sep), query_folder)
    corresps_path = os.path.join(path, 'gt_corresps.pkl')

    with open(corresps_path, 'rb') as f:
        actual_corresps = pickle.load(f)
    return actual_corresps

def load_mask_images(query_folder):
    """
    Loads all the mask images stored on the query folder
    Args:
        - query_folder: specifies the query_folder name
    Returns:
        A list of Image instnaces
    """

    path = os.path.join(os.path.dirname(__file__),
                        '../{}'.format(query_folder))
    mask_paintings = []
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.png':
            mask_painting = Painting(os.path.join(path, filename))
            mask_paintings.append(mask_painting)
    return mask_paintings

"""
TODO: USe the new background remover
def remove_background_and_store(images_with_background, folder):
    images_without_background = []
    masks = []

    remover = BasicRemovingStrategy()

    path = os.path.join(os.path.dirname(__file__), '../{}'.format(folder))

    for image in images_with_background:
        images_without_background.append(remover.remove_background(image, 10, 10))
        masks.append(remover.mask)
        filename = image.filename + '.png'
        filename = os.path.join(path, filename)
        remover.store_mask(filename)
    return images_without_background, masks
"""

"""
TODO: Check again the 3D Histo
def calc_3d_histogram(images, hist_size=[256], ranges=[0, 256], mask=None):

    histograms = []
    for image in images:
        red_histogram = image.calc_histogram(2, hist_size, ranges, mask)
        cv2.normalize(red_histogram, red_histogram, norm_type=cv2.NORM_MINMAX)

        green_histogram = image.calc_histogram(1, hist_size, ranges, mask)
        cv2.normalize(green_histogram, green_histogram, norm_type=cv2.NORM_MINMAX)

        blue_histogram = image.calc_histogram(0, hist_size, ranges, mask)
        cv2.normalize(blue_histogram, blue_histogram, norm_type=cv2.NORM_MINMAX)

        histogram = red_histogram + green_histogram + blue_histogram
        histograms.append(histogram)
    return histograms
"""