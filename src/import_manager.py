from pathlib import Path
from museum_item import MuseumItem
import os
import pickle
from cv2 import cv2
from painting import Painting

def import_bbdd_paintings():
    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    paintings = []
    for filename in sorted(os.listdir(path)):
        painting = Painting(os.path.join(path, filename))
        paintings.append(painting)
    return paintings

def import_all_bbdd_museum_items(query_folder):
    path = os.path.join(os.path.dirname(__file__), '../data/{}'.format(query_folder))
    museum_items = []
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.pkl':
            with open('{}/{}'.format(path, filename), 'rb') as f:
                museum_item = pickle.load(f)
                museum_items.append(museum_item)
    return museum_items

def import_query_paintings(query_folder):
    query_paintings = []
    path = os.path.join(os.path.dirname(__file__), '../queries/{}'.format(query_folder))
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.jpg':
            painting = Painting(os.path.join(path, filename))
            query_paintings.append(painting)
    return query_paintings

def import_gt_corresps(query_folder):
    path = os.path.join(os.path.dirname(__file__), '../queries/{}'.format(query_folder))
    with open('{}/{}'.format(path, 'gt_corresps.pkl'), 'rb') as gt_f:
        return pickle.load(gt_f)

def import_text_boxes(query_folder):
    path = os.path.join(os.path.dirname(__file__), '../queries/{}'.format(query_folder))
    with open('{}/{}'.format(path, 'text_boxes.pkl'), 'rb') as f:
        return pickle.load(f)