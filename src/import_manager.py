from pathlib import Path
import pickle
import os

def import_bbdd_image_paths():
    bbdd_image_paths = []
    path = os.path.join(os.path.dirname(__file__), '../bbdd')
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.jpg':
            bbdd_image_paths.append('{}/{}'.format(path,filename))
    return bbdd_image_paths

def import_query_image_paths(query_folder):
    query_image_paths = []
    path = os.path.join(os.path.dirname(__file__), '../queries/{}'.format(query_folder))
    for filename in sorted(os.listdir(path)):
        if Path(filename).suffix == '.jpg':
            query_image_paths.append('{}/{}'.format(path,filename))
    return query_image_paths

def import_gt_corresps(query_folder):
    path = os.path.join(os.path.dirname(__file__), '../queries/{}/gt_corresps.pkl'.format(query_folder))
    with open(path, 'rb') as gt:
        return pickle.load(gt)    