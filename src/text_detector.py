import  cv2
import matplotlib.pyplot as plt
from skimage.morphology import square
import skimage.segmentation

def erase_overlapped_regions(rects):
    """ 
    Remove the overlapped rectangles and merge the ones that are very near
    """
    num_rects = len(rects)-1
    ext_rects = rects[:]
    for i in range(num_rects):
        for j in range(i+1, len(rects)):
            
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            
            if x2 > x1 and x2 < x1+w1 and y2 > y1 and y2 < y1+h1:
                #x1 contains x2
                ext_rects.remove(rects[j])
            elif x1 > x2 and x1 < x2+w2 and y1 > y2 and y1 < y2+h2:
                #x2 contains x1
                ext_rects.remove(rects[i])
            else:
                if x2>x1 and x2-x1-w1<15:
                    #boxes are very near or overlapped
                    x3=x1
                    y3=y1
                    w3=x2+w2-x1
                    h3=h1
                    ext_rects.remove(rects[i])
                    ext_rects.remove(rects[j])
                    new_rect = (x3, y3, w3, h3)
                    ext_rects.append(new_rect)
                elif x1>x2 and x1-x2-w2<15:
                    #boxes are very near or overlapped
                    x3=x2
                    y3=y1
                    w3=x1+w1-x2
                    h3=h1
                    ext_rects.remove(rects[i])
                    ext_rects.remove(rects[j])
                    new_rect = (x3, y3, w3, h3)
                    ext_rects.append(new_rect)
                else:
                    pass
    return ext_rects

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
    
    im_x, im_y = gray_im.shape
    im_area =im_x*im_y
    rects = []
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        rect_area = w*h
        ratio = h/w

        if w >= 60 and w*0.60 > h and 0.0015*im_area<rect_area<0.20*im_area and ratio>0.08 and h>60:
            """
             if w and h aren't too small,
                area of the rectangle isn't too small or big respect the image area
                and ratio between h/w is bigger than 0.08 and smaller than 0.6
            we append the rectangle 
             """
            rect = (x, y, w, h)
            rects.append(rect)

    no_overlapped_rects = erase_overlapped_regions(rects)
            
        
    for rectangle in no_overlapped_rects:
        x, y, w, h = rectangle
        cv2.rectangle(im_rgb, (x, y), (x+w, y+h), (255, 0, 0), 5);
            
    if plot_results:
        fig= plt.figure(figsize=(11,15))
        plt.imshow(im_rgb)

    return no_overlapped_rects
    
# path = 'D:\\Users\\USUARIO\\Documents\\M1.IntroductionToHumanAndVC\\M1.P2\\qsd1_w2\\00005.jpg'
# image = cv2.imread(path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# rectangle_box = detect_text_box(gray, True)
