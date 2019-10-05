from . import pipeline

"""
This file is for calc the images offline
"""

def run():
    images = pipeline.load_bbdd_images()
    pipeline.apply_change_of_color_space(images, 'GRAY')
    images_histograms = pipeline.calc_image_histogram(images, 0)
    pipeline.export(images, images_histograms)