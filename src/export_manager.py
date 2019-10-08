from pathlib import Path
import os
from painting import Painting
from museum_item import MuseumItem
import pickle


def export_museum_item(museum_item):
    """
    Stores an museum item object into data
    Args:
        - museum_item: is an instance of a MuseumItem class
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    with open('{}/{}.pkl'.format(path, museum_item.painting.filename),
              'wb') as file_obj:
        pickle.dump(museum_item, file_obj)
