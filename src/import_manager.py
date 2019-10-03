from pathlib import Path
import os
from image import Image
import pickle


def import_image(filename):
    """
    Export an image into JSON file stored on export folder. Add a new entry
    if that image does not exists, otherwise it will overwrite it
    Args:
        - filename: is the name of the filename with an image stored
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    with open('{}/{}.pkl'.format(path, filename), 'rb') as file_obj:
        return pickle.load(file_obj)
    