import cv2.cv2 as cv2
import numpy as np
from image import Image
from scipy.signal import find_peaks
import os
import glob


class BackgroundRemoverBase:
    def __init__(self):
        self.mask = None

    def save_mask(self, mask_filename):
        cv2.imwrite(mask_filename, self.mask)


class BasicRemovingStrategy(BackgroundRemoverBase):
    def __init__(self, height_ratio=10, distance_between_peaks=10):
        super().__init__()
        self.height_ratio = height_ratio
        self.distance_between_peaks = distance_between_peaks

    def remove_background(self, image, height_ratio, distance_between_peaks):
        """
            TODO
        """
        def my_find_peaks(array, height_ratio, distance_between_peaks):
            peaks, _ = find_peaks(array,
                                  height=np.max(array) / self.height_ratio,
                                  distance=self.distance_between_peaks)
            return peaks

        gray_image = cv2.cvtColor(image.img, cv2.COLOR_BGR2GRAY)

        col_var = np.abs(np.gradient(gray_image.mean(0)))

        left_cd = my_find_peaks(col_var[:len(col_var) // 2], height_ratio,
                                distance_between_peaks)
        left = left_cd[0]

        right_cd = my_find_peaks(col_var[len(col_var) // 2:], height_ratio,
                                 distance_between_peaks) + len(col_var) // 2
        right = right_cd[-1]

        row_var = np.abs(np.gradient(gray_image.mean(1)))

        top_cd = my_find_peaks(row_var[:len(row_var) // 2], height_ratio,
                               distance_between_peaks)
        top = top_cd[0]

        bottom_cd = my_find_peaks(row_var[len(row_var) // 2:], height_ratio,
                                  distance_between_peaks) + len(row_var) // 2
        bottom = bottom_cd[-1]

        self.mask = np.zeros(shape=(image.img.shape[0], image.img.shape[1]),
                             dtype=np.uint8)
        self.mask[top:bottom, left:right] = 255
        cropped_image = image.img[top:bottom, left:right].copy()
        return cropped_image
