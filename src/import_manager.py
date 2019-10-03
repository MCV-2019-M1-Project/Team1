from pathlib import Path
import os
import pickle


def import_museum_item(filename):
    """
    Import a museum item previously stored on data.
    Args:
        - filename: is the name of the museum item filename stored
    """

    path = os.path.join(os.path.dirname(__file__), '../data')
    with open('{}/{}.pkl'.format(path, filename), 'rb') as file_obj:
        return pickle.load(file_obj)
