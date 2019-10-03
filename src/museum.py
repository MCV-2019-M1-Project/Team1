from image import Image
from export_manager import export_museum_item
from import_manager import import_museum_item
from museum_item import MuseumItem
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

museum_item = MuseumItem(img)
export_museum_item(museum_item)

museum_item = import_museum_item(img.filename)
print('Img filename restored', museum_item.image.filename)