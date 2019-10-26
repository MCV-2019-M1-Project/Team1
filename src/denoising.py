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
    #TODO: try with openings and closings
    return image_filtered

def remove_hue_variation(image, mean_hue):
    """
    Standarizes the hue of the image
    Args:
        - image: image in BGR color space
    Returns
        Image in BGR color space with standarized hue
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = image[:,:,0]
    mean = np.mean(hue)
    shift = mean_hue - mean
    new_hue = (hue+shift)%180
    image_filtered = image
    image_filtered[:,:,0] = new_hue
    image_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_HSV2BGR)
    return image_filtered
