import  cv2
import matplotlib.pyplot as plt
from skimage.morphology import square
import skimage.segmentation


def detect_text_box(gray_im, plot_results):
    """
    Detect the bounding box of the text

    args:
    - gray_im: image in grayscale 
    - plot_results: if true imshow of the orginal gray image with the bounding box contour superposed
    output:
    - rects: Coordinates, width and height 
    """


    #Disk Structural elements
    S5 = skimage.morphology.disk(5)

    S3 = skimage.morphology.disk(3)
    
    
    #Applying closing to the image
    closing = cv2.morphologyEx(gray_im, cv2.MORPH_CLOSE, S5)
    
    #Applying opening to the image
    opening = cv2.morphologyEx(gray_im, cv2.MORPH_OPEN, S5)
       
    #difference between closing of the image and opening
    Im_op_clo_diff = closing - opening
    #Gaussian filter to smooth the image and reduce noise for a better binarization
    blur_img = cv2.GaussianBlur(Im_op_clo_diff,(5,5),3)
    
    #Binarization of the blurred image
    ret,th3 = cv2.threshold(blur_img,2,255,cv2.THRESH_BINARY)
    
    #dilation of the binarized image to reduce black areas
    dilation = cv2.dilate(th3, S3,2)

    #find contours of the connected components in the binarized image
    im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    im_rgb = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2RGB)
    #g=cv2.drawContours(im_rgb, contours[0:], -1, (0, 0, 255), 9)
    #plt.imshow(g);
    
    rects = []
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        if w >= 60 and w > h:
            # if width is bigger than height and isn't too small
            rect = (x, y, w, h)
            rects.append(rect)
            cv2.rectangle(im_rgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
    if plot_results:
        fig= plt.figure(figsize=(11,15))
        plt.imshow(im_rgb)

    return rects
    

# path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P2\\qsd1_w2\\00024.jpg'
# image = cv2.imread(path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# rectangle_box = detect_text_box(gray)