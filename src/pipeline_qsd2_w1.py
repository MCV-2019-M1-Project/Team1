from pipeline import (
    load_query_images,
    calc_image_histogram,
    calc_similarty,
    apply_change_of_color_space,
    load_mask_images,
    remove_background_and_store,
    load_gt_corresps,
    calc_3d_histogram
)
from import_manager import import_all_museum_items
from image import Image
from museum_item import MuseumItem
from k_retrieval import k_retrieval
from evaluation_metrics import global_compute_image_comparision, global_mapk

def run(k=10):
    #Load items
    query_images = load_query_images('qsd2_w1')
    bbdd_museum_items = import_all_museum_items()
    gt_corresps = load_gt_corresps('qsd2_w1')
    mask_images = load_mask_images('qsd2_w1')

    #Remove backgound
    images_without_background, predicted_masks = remove_background_and_store(query_images, 'week1/QST2/method1/')
    
    #Compare images to museum item and obtain metrics
    apply_change_of_color_space(images_without_background, 'YCrCb')
    query_histograms = calc_3d_histogram(images_without_background)
    query_museum_items = []
    for image, histogram in zip(images_without_background, query_histograms):
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

    #Background metrics
    precision, recall, f1 = global_compute_image_comparision(mask_images, predicted_masks)
    print("precision = ", precision)
    print("recall = ", recall)
    print("f1 score = ", f1)

    return output
