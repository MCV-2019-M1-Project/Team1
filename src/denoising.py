from cv2 import cv2
import numpy as np

def denoise_image(image, mean_hue):
    """
    Denoises an image
    Args:
        - image: image in BGR color space
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
        - image: image in BGR color space
    Returns
        Image without the salt and pepper noise
    """

    #img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Using median filter
    kernel = 3
    image_filtered = cv2.medianBlur(image,kernel)
    return image_filtered

def remove_hue_variation(image, mean_hue):
    """
    Standarizes the hue of the image
    Args:
        - image: image in BGR color space
    Returns
        Image with standarized hue
    """

    image_filtered = image
    return image_filtered
