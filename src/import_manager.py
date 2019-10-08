from pathlib import Path
from museum_item import MuseumItem
import os
import pickle


def import_museum_item(filename):
    """
    Import a museum item previously stored on data folder.
    Args:
        - filename: is the name of the museum item filename stored
    Returns:
        A museum item related to the filename
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    with open('{}/{}'.format(path, filename), 'rb') as file_obj:
        return pickle.load(file_obj)


def import_all_museum_items():
    """
    Import all the museum items previously stored on data folder
    Returns:
        All the museum items
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    museum_items = []
    for filename in os.listdir(path):
        if Path(filename).suffix == '.pkl':
            museum_items.append(import_museum_item(filename))
    return museum_items
