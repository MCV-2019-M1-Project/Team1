import math
import numpy as np
try:
    import cv2
except ImportError:
    import cv2.cv2 as cv2
from skimage import data, feature, exposure
import matplotlib.pyplot as plt


def laplacian_of_gaussians(sigma, size):
    """
    Creates a LoG kernel for the convolution with the image 
    - size: wanted size for the kernel
    - sigma: parameter to compute LoG
    """
    log_kernel = np.zeros((size, size))

    for i in range(-int(size/2), int(size/2)):
        for j in range(-int(size/2),int(size/2)):
            log_kernel[i][j] = (-1)/(math.pi*sigma**4)*(1-(i**2+j**2)/(2*sigma**2))*math.exp(-(i**2+j**2)/(2*sigma**2))
    
    return log_kernel

def LoG_convolve(img, filt_size, k,sigma):
    """
    Computes the convolutions of the input image with different LoG kernels using:
    - filt_size as kernel size
    - sigma and k parameters for computing the diferent LoG
    """
    log_images = [] #to store responses
    for i in range(0,9):
        y = np.power(k,i) 
        sigma_1 = 1*y #sigma 
        filter_log = laplacian_of_gaussians(sigma_1, filt_size) #filter generation
        image = cv2.filter2D(img,-1,filter_log) # convolving image
        image = np.pad(image,((1,1),(1,1)),'constant') #padding 
        image = np.square(image) # squaring the response
        log_images.append(image)
    log_image_np = np.array([i for i in log_images]) 
    
    return log_image_np


def detect_blob(img, log_image_np, k, sigma, thr = 0.02):
    coordinates = [] 
    (h,w) = img.shape
    for i in range(1,h):
        for j in range(1,w):
            slice_img = log_image_np[:,i-1:i+2,j-1:j+2] #size*3*3 slice
            result = np.amax(slice_img) #finding maximum
            if result >= thr: #threshold
                z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
                coordinates.append((i+x-1,j+y-1,k**z*sigma)) #finding co-rdinates
    return coordinates

def compute_blob_keypoints(gray_img, k = 1.414, sigma = 1.0, filt_size = 7):
    """
    Compute image blobs with LoG (Laplacian of Gaussians) convolution with the image:
    - k and sigma are parameters to compute the LoG
    - filt_size is the size of the kernel computed to convolve with the img
    
    Returns a list of keypoints as keypoin object
    The Laplacian archives maximum response for the binary circle of radius r is at σ=1.414*r
    """
    norm_img = gray_img/255.0 #image normalization
    log_image_np = LoG_convolve(norm_img, filt_size, k, sigma)
    co_ordinates = list(set(detect_blob(norm_img, log_image_np, k, sigma)))
    nh,nw = norm_img.shape
    keyPoints = []
    for blob in co_ordinates:
        y,x,r = blob
        keyPoints.append(cv2.KeyPoint(x, y, r*k))
    return keyPoints


###EXECUTION EX:

# gray_query_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# blob_kp_query = compute_blob_keypoints(gray_query_img) #Returns a list of keypoints as keypoints objects
