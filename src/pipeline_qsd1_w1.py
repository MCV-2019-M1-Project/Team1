from pipeline import (
    load_query_images,
    load_bbdd_images,
    calc_image_histogram,
    calc_similarty,
    apply_change_of_color_space,
    calc_3d_histogram,
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

    apply_change_of_color_space(query_images, 'YCrCb')
    query_histograms = calc_3d_histogram(query_images)
    query_museum_items = []
    for image, histogram in zip(query_images, query_histograms):
        query_museum_item = MuseumItem(image, histogram)
        query_museum_items.append(query_museum_item)

    distances = []
    for query_museum_item in query_museum_items:
        distance = calc_similarty(bbdd_museum_items, query_museum_item, 'hellinger')        
        distances.append(distance)

    print('Score: ', global_mapk(gt_corresps, distances, k))

    output = []
    for distance in distances:
        k_distances = (k_retrieval(distance, k))
        output_list = []
        for k_distance in k_distances:
            output_list.append(k_distance.db_im.image.image_name())
        output.append(output_list)

    # Print results
    for gt_corresp, predicted in zip(gt_corresps, output):
        print('Solution (GT): ', gt_corresp)
        print('Predicted: ', predicted)

    return output