from pathlib import Path
import os
from image import Image
import pickle


def export_image(image):
    """
    Export an image into JSON file stored on export folder. Add a new entry
    if that image does not exists, otherwise it will overwrite it
    Args:
        - image: is an instance of a Image class
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    with open('{}/{}.pkl'.format(path,image.filename), 'wb') as file_obj:
        pickle.dump(image, file_obj)