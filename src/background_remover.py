import cv2.cv2 as cv2
import numpy as np
from painting import Painting
from scipy.signal import find_peaks

def remove_background(painting, height_ratio, distance_between_peaks):
    """
        TODO
    """

    def my_find_peaks(array, height_ratio, distance_between_peaks):
        peaks, _ = find_peaks(array,
                                height=np.max(array) / height_ratio,
                                distance=distance_between_peaks)
        return peaks

    gray_image = cv2.cvtColor(painting.img, cv2.COLOR_BGR2GRAY)

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

    mask = np.zeros(shape=(painting.img.shape[0], painting.img.shape[1]),
                            dtype=np.uint8)
    painting.mask[top:bottom, left:right] = 255
    
    cropped_image = painting
    cropped_image.img = painting.img[top:bottom, left:right].copy()

    return cropped_image
