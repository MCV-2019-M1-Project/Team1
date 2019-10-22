import cv2
import pytesseract
import matplotlib.pyplot as plt
from text_detector import detect_text_box
import difflib
import os.path as path

two_up =  path.abspath(path.join(__file__ ,"../.."))
pth = path.join(two_up ,"Tesseract-OCR\\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = pth

def recognize_text(image, config='-l eng --oem 1 --psm 3'):
    """
    From an input image
    Detects the text region
    Performs a binarization of it 
    Returns the text contained in the binarized image
    """
    txt_bx = detect_text_box(image, False)
    txt_im = image[txt_bx[1]:txt_bx[3]+1, txt_bx[0]:txt_bx[2]+1]
    ret,th3 = cv2.threshold(txt_im, 160, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(th3, config=config)
    plt.imshow(th3, 'gray') 
    return text

def image_text_retrieval(text, text_list):
    """
    Returns the index of the closest string in the text_list of the input text
    """
    
    closest_match = difflib.get_close_matches(text, text_list, cutoff=0.35)
    match_index = text_list.index(closest_match[0])
    
    return match_index
    