from pipeline import (
    load_bbdd_images,
    apply_change_of_color_space,
    calc_image_histogram,
    calc_3d_histogram,
    export
)
"""
This file is for calc the images offline
"""

def run():
    images = load_bbdd_images()
    apply_change_of_color_space(images, 'LAB')
    images_histograms = calc_3d_histogram(images)
    export(images, images_histograms)
