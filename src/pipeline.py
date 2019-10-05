from pathlib import Path
import os
from distance import Distance
from export_manager import export_museum_item
from museum_item import MuseumItem
from image import Image
import pickle


def load_bbdd_images():
    """
    Load all the images stored on the bbdd folder
    Returns:
        A list of Image object instances
    """

    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    images = []
    for filename in os.listdir(path):
        image = Image(os.path.join(path, filename))
        images.append(image)
    return images


def load_query_images(query_folder):
    """
    Loads all the query images stored on the query folder
    Args:
        - query_folder: specifies the query_folder name
    Returns:
        A list of Image instnaces
    """

    path = os.path.join(os.path.dirname(__file__),
                        '../{}'.format(query_folder))
    query_images = []
    for filename in os.listdir(path):
        if Path(filename).suffix == '.jpg':
            query_image = Image(os.path.join(path, filename))
            query_images.append(query_image)
    return query_images


def apply_change_of_color_space(images, color_space):
    """
    Apply to all images a color space change
    Args:
        - images: a list of Image object instances
        - color_space: the color space to change could be
            * BGR
            * HSV
            * LAB
            * GRAY
    """

    for image in images:
        image.color_space = color_space


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


def calc_image_equalize_hist(images):
    """
    Do an image processing of contrast adjustment using the image's histogram
    Args:
        - images: a list of Image object instances
    Returns:
        A list of equalized histograms based on the given images
    """

    equalize_histograms = []
    for image in images:
        equalize_histogram = image.calc_equalize_hist()
        equalize_histograms.append(equalize_histogram)
    return equalize_histograms


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


def calc_similarty(db_museum_items, query_museum_item, method):
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
        distance = Distance(db_museum_item, query_museum_item)
        distance.calc_dist(method)
        print(distance.distance)
        distances.append(distance)
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
    mask_images = []
    for filename in os.listdir(path):
        if Path(filename).suffix == '.png':
            mask_images = Image(os.path.join(path, filename))
            mask_images.append(query_image)
    return mask_images
