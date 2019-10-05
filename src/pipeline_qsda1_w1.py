from . import evaluation_metrics
from . import pipeline
from . import evaluation_metrics
from import_manager import import_all_museum_items


def run():
    query_images = pipeline.load_query_images('qsd1_w1')
    museum_items = import_all_museum_items()

    pipeline.apply_change_of_color_space(query_images, 'GRAY')
    query_histograms = pipeline.calc_image_histogram(query_images, 0)

    similarities_matrix = []
    for query_histogram in query_histograms:
        similarities_matrix.append(
            pipeline.calc_similarty(museum_items, query_histogram,
                                    'euclidean'))

    # What TODO now
    evaluation_metrics.mapk()
