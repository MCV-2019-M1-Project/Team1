import cv2.cv2 as cv2
import numpy as np
from scipy.signal import find_peaks
import os
import glob

class BackgroundRemoverBase:

    def __init__(self):
        self.mask = None

    def save_mask(self, mask_filename):
        cv2.imwrite(mask_filename, self.mask)


class BasicRemovingStrategy(BackgroundRemoverBase):

    def __init__(self,
                 height_ratio=10,
                 distance_between_peaks=10):
        super().__init__()
        self.height_ratio = height_ratio
        self.distance_between_peaks = distance_between_peaks

    def remove_background(self, numpy_image):
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        col_var = np.abs(np.gradient(gray.mean(0)))

        left_cd = self._my_find_peaks(col_var[:len(col_var) // 2])
        left = left_cd[0]

        right_cd = self._my_find_peaks(col_var[len(col_var) // 2:]) + len(col_var) // 2
        right = right_cd[-1]

        row_var = np.abs(np.gradient(gray.mean(1)))

        top_cd = self._my_find_peaks(row_var[:len(row_var) // 2])
        top = top_cd[0]

        bottom_cd = self._my_find_peaks(row_var[len(row_var) // 2:]) + len(row_var) // 2
        bottom = bottom_cd[-1]

        self.mask = np.zeros(shape=(numpy_image.shape[0], numpy_image.shape[1]), dtype=np.uint8)
        self.mask[top:bottom, left:right] = 255
        cropped_image = numpy_image[top:bottom, left:right].copy()
        return cropped_image