from image import Image
from histogram import Histogram
import cv2

# Testing the image class
img = Image('bbdd/bbdd_00000.jpg')
print(img.img)
histogram = Histogram.calc_histogram(img, 0)
print(histogram)
equalize = Histogram.calc_equalize_hist(img)
print(equalize)