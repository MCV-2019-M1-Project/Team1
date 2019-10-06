from pipeline import (
    load_query_images,
    load_bbdd_images,
    calc_image_histogram,
    calc_similarty,
    apply_change_of_color_space,
    load_gt_corresps
)
from import_manager import import_all_museum_items
from museum_item import MuseumItem
from k_retrieval import k_retrieval
from evaluation_metrics import (
    mapk,
    global_mapk
)

def run(k=10):
    query_images = load_query_images('qsd1_w1')
    bbdd_museum_items = import_all_museum_items()
    gt_corresps = load_gt_corresps('qsd1_w1')

    apply_change_of_color_space(query_images, 'GRAY')
    query_histograms = calc_image_histogram(query_images, 0)
    
    query_museum_items = []
    for image, histogram in zip(query_images, query_histograms):
        query_museum_item = MuseumItem(image, histogram)
        query_museum_items.append(query_museum_item)

    distances = []
    for query_museum_item in query_museum_items:
        distance = calc_similarty(bbdd_museum_items, query_museum_item, 'euclidean')        
        distances.append(distance)

    global_mapk(gt_corresps, distances, k)