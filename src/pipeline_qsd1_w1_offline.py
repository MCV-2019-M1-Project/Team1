from pipeline import (
    load_bbdd_images,
    apply_change_of_color_space,
    calc_image_histogram,
    export
)
"""
This file is for calc the images offline
"""

def run():
    images = load_bbdd_images()
    apply_change_of_color_space(images, 'GRAY')
    images_histograms = calc_image_histogram(images, 0)
    export(images, images_histograms)
