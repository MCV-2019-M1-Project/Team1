from pipeline import (
    load_query_images,
    calc_image_histogram,
    calc_similarty,
    apply_change_of_color_space
    load_mask_images
    remove_background
)
from import_manager import import_all_museum_items
from museum_item import MuseumItem
from k_retrieval import k_retrieval
from evaluation_metrics import global_compute_image_comparision

def run():
    #Load items
    query_images = load_query_images('qsd2_w1')
    bbdd_museum_items = import_all_museum_items()
    actual_correspondances = load_gt_correspondances('qsd2_w1')
    mask_images = load_mask_images('qsd2_w1')

    #Remove backgound
    images_without_background, predicted_masks = remove_background(query_images)

    #TODO: pipeline equal to qsd1

    #Background metrics
    precision, recall, f1 = global_compute_image_comparision(mask_images, predicted_masks):
    print("precision = ", precision)
    print("recall = ", recall)
    print("f1 score = ", f1)
