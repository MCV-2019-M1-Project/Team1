from cv2 import cv2
import numpy as np


def denoise_image(image, mean_hue):
    """
    Denoises an image
    Args:
        - image: image that wants to be denoised
    Returns
        Denoised image
    """

    image_filtered1 = remove_salt_and_pepper_noise(image)
    image_filtered2 = remove_hue_variation(image_filtered1, mean_hue)
    return image_filtered2

def remove_salt_and_pepper_noise(image):
    """
    Removes the salt and pepper noise of an image
    Args:
        - image: image that wants to be denoised
    Returns
        Image without the salt and pepper noise
    """

    image_filtered = image
    return image_filtered

def remove_hue_variation(image, mean_hue):
    """
    Standarizes the hue of the image
    Args:
        - image
    Returns
        Image with standarized hue
    """

    image_filtered = image
    return image_filtered
