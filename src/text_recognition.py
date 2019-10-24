import cv2
import pytesseract
import matplotlib.pyplot as plt
from text_detector import detect_text_box
import difflib
import os.path as path
import platform
from background_remover import get_bbox

if platform.system() == 'Windows':
    two_up =  path.abspath(path.join(__file__ ,"../.."))
    pth = path.join(two_up ,"Tesseract-OCR\\tesseract.exe")
    pytesseract.pytesseract.tesseract_cmd = pth

def text_recognition(image, config = '-l eng --oem 1 --psm 3',plot_rect = False):
    """
    From an input image ()
    Detects the text region
    Performs a binarization of it 
    Returns the text contained in the binarized image
    """
    txt_bx = detect_text_box(image, False)
    txt_im = image[txt_bx[1]:txt_bx[3]+1, txt_bx[0]:txt_bx[2]+1]
    ret,th3 = cv2.threshold(txt_im, 160, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(th3, config=config)
    if plot_rect:
        fig= plt.figure(figsize=(11,15))
        plt.imshow(th3, 'gray') 
    
    return text

def save_single_text(filename, text, mode):
    """
    Open a file "filename" using the input mode and save input text
    """
    f = open(filename + ".txt", mode)
    f.write(text)
    
def image_text_retrieval(text, text_list):
    """
    Returns the index of the closest string in the text_list of the input text
    """
    
    closest_match = difflib.get_close_matches(text, text_list, cutoff=0.35)
    match_index = text_list.index(closest_match[0])
    
    return match_index

def save_image_text(filename, paints, gray_img, recog_config):
    """
    From a list of coordinates of the paintings in an image,
    computes the text detection and recognition and saves the resultant text in a file for every image
    """
    for p in paints:
        obtained_text = text_recognition(gray_img[p[1]:p[3], p[0]:p[2]], recog_config, False)
        if paintings.index(p)==0:
            save_single_text(filename, obtained_text, "w+")
        if paintings.index(p)!=0:
            save_single_text(filename, obtained_text, "a")

##### Ex   
#config = ('-l eng --oem 1 --psm 3')
#
#for i in range(29, -1, -1):
#    n = str(i).zfill(2)
#    path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P3\\qsd2_w3\\000%s.jpg' % (n)
##path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P3\\qsd2_w3\\00024.jpg'
#    image = cv2.imread(path)
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    paintings = []
#    paintings = get_bbox(image)
#    save_image_text('000%s' % (n), paintings, gray, config)

