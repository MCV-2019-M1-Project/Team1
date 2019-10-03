from image import Image
from export_manager import export_image
from import_manager import import_image
import cv2

# Testing the image class
img = Image('bbdd/bbdd_00000.jpg')
print('Img filename', img.filename)
print('Img color space', img.color_space)
print('Img the img itself', img.img)
print('Get the basic hist', img.calc_histogram(0))
print('Get the equalized hist', img.calc_equalize_hist())
print('change to HSV color space')
img.color_space = 'HSV'
print('Img color space', img.color_space)

export_image(img)

img = import_image(img.filename)
print('Img filename restored', img.filename)