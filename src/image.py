from pathlib import Path
from cv2 import cv2


class Image(object):
    def __init__(self, filename):
        self.set_img(filename)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    @property
    def filename(self):
        return self._filename
   
    def set_img(self, filename):
        self._color_space = 'BGR'
        self._filename = int(filename.split('/')[-1:][0][-9:-4])
        self._img = cv2.imread(filename)

    def set_color_space(self, new_color_space):
        if self._color_space == 'BGR':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2HSV)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)
            elif new_color_space == 'GRAY':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            elif new_color_space == 'YCrCb':                
                self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2YCrCb)
            else:
                raise NotImplementedError

        elif self._color_space == 'HSV':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2BGR)
            elif new_color_space == 'LAB':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_HSV2LAB)
            else:
                raise NotImplementedError

        elif self._color_space == 'LAB':
            if new_color_space == 'HSV':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2HSV)
            elif new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)
            else:
                raise NotImplementedError

        elif self._color_space == 'GRAY':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)
            else:
                raise NotImplementedError
        elif self._color_space == 'YCrCb':
            if new_color_space == 'BGR':
                self._img = cv2.cvtColor(self._img, cv2.COLOR_YCrCb2BGR)
            else:
                raise NotImplementedError        
        else:
            raise NotImplementedError

        self._color_space = new_color_space