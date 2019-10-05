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



def run():
    k = 10
    query_images = load_query_images('qsd1_w1')
    bbdd_images = load_bbdd_images()
    actual_correspondances = load_gt_corresps('qsd1_w1')

    apply_change_of_color_space(query_images, 'GRAY')
    query_histograms = calc_image_histogram(query_images, 0)
    bbdd_histograms = calc_image_histogram(bbdd_images, 0)
    
    query_museum_items = []
    for image, histogram in zip(query_images, query_histograms):
        query_museum_item = MuseumItem(image, histogram)
        query_museum_items.append(query_museum_item)
        
    bbdd_museum_items = []
    for image, histogram in zip(bbdd_images, bbdd_histograms):
        bbdd_museum_item = MuseumItem(image, histogram)
        bbdd_museum_items.append(bbdd_museum_item)
                                  
    distances = calc_similarty(bbdd_museum_items, query_museum_item, 'correlation')
    k_retrieved = k_retrieval(distances, k)

    mapk_list = []
    for act_corresp in actual_correspondances:
        mapk_list.append(mapk(k, act_corresp, k_retrieved))
    mapk_mean = global_mapk(k, query_images, act_corresp, k_retrieved)

    return k_retrieved, mapk_mean



run()



    
    

    