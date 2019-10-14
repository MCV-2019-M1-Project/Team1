import  cv2
import matplotlib.pyplot as plt
from skimage.morphology import square
import skimage.segmentation
import itertools

def remove_overlapped(over_Rects, extt_rects, ind1, ind2):
        """
        Remove the rectangles that are overlapped and append a new rectangle that contains both
        """
        x1, y1, w1, h1 = extt_rects[ind1]
        x2, y2, w2, h2 = over_Rects
        x3=min(x1,x2)
        y3=min(y1,y2)
        w3=max(x2+w2-x1, x1+w1-x2) 
        h3=max(h1,h2)
        elem1=extt_rects[ind1]
        elem2=extt_rects[ind2]
        extt_rects.remove(elem1)
        extt_rects.remove(elem2)
        new_rect = (x3, y3, w3, h3)
        extt_rects.append(new_rect)
        
        return extt_rects
    
def erase_inside_regions(rects):
    """ 
    Remove the rectangles contained in others
    """
    num_rects = len(rects)-1
    ext_rects = rects[:]
    contains = []
    for i in range(num_rects+1):
        contains[:] = []
        for j in range(num_rects+1):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            
            if (x2 > x1 and x2+w2 < x1+w1 and y2 > y1 and y2+h2 < y1+h1):
                #x1 contains x1

                contains.append(1)
               
            else:
                contains.append(-1)


        contained_list =[item == -1 for item in contains]
        ext_rects = list(itertools.compress(ext_rects, contained_list))
        return ext_rects
    
def erase_overlapped_regions(rects, im_w, im_h):
    
    """ 
    Remove the overlapped rectangles and merge the ones that are very near
    """

    index_list=[]
    ext_rects = rects[:]
    overlapped = []
    for i in ext_rects:
        overlapped[:] = []
        for j in ext_rects:
            x1, y1, w1, h1 = i
            x2, y2, w2, h2 = j
            
            if x1>x2 and abs(x1-x2-w2)<0.005*im_w and -0.005*im_w<y1-y2<0.005*im_h:
                overlapped.append(ext_rects.index(j))
            else:
                overlapped.append(-1)
                
        index_list[:]=[]
              
        overlapped_list = [item != -1 for item in overlapped]
        elem_overlapped = list(itertools.compress(ext_rects, overlapped_list))   
        for el in overlapped:
            if el != -1:
                index_list.append(el)
            else:
                pass
        for it in range(len(elem_overlapped)):
            ext_rects = remove_overlapped(elem_overlapped[it], ext_rects, ext_rects.index(i), index_list[it])

        
        
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

        if w >= 0.02*im_x and w*0.50 > h and h>=0.02*im_y and 0.0020*im_area<rect_area<0.20*im_area:
            """
             if w and h aren't too small,
                area of the rectangle isn't too small or big respect the image area
                and ratio between h/w is bigger than 0.08 and smaller than 0.6
            we append the rectangle 
             """
            rect = (x, y, w, h)
            rects.append(rect)
       


    outside_rects = erase_inside_regions(rects)
    no_overlapped_rects = erase_overlapped_regions(outside_rects, im_x, im_y)

    detected_rects = []        
    list_dilation = [] 
    


    for rectangle in no_overlapped_rects:
        x, y, w, h = rectangle
        # List of rectangles to return the coordinates as required (tlx, tly, brx, bry)
        det_rect = (x, y, x+w, y+h)
        detected_rects.append(det_rect)
        list_dilation.append(sum(sum(dilation[y:(y+h) , x:x+w]))) 
    #take the rectangle with biggest values inside    
    max_dilation_rect=max(list_dilation)
    text_rect = detected_rects[list_dilation.index(max_dilation_rect)]
    cv2.rectangle(im_rgb, (text_rect[0], text_rect[1]), (text_rect[2], text_rect[3]), (255, 0, 0), 5)
      
    if plot_results:
        fig= plt.figure(figsize=(11,15))
        plt.imshow(im_rgb)  

    return detected_rects