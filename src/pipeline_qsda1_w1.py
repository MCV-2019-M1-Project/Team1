
from . import pipeline
from import_manager import import_all_museum_items

def run():
    query_images = pipeline.load_query_images('qsd1_w1')
    museum_items = import_all_museum_items()

    pipeline.apply_change_of_color_space(query_images, 'GRAY')
    query_histograms = pipeline.calc_image_histogram(query_images, 0)

    similarities = 
    for query_histogram in query_histograms:
        pipeline.calc_similarty(museum_items, query_histogram)
    
    
