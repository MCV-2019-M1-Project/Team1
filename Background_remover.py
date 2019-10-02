import numpy as np


class Background_remover:

    def __init__(self, offline=False):
        self.offline = offline
        pass

    def remove_background(self, numpy_image, mask_filename=None):
        #TODO: Find edges

        self.mask = None

        if self.offline:
            self.save_mask(mask_filename)

        return

    def save_mask(self, mask_filename=None):
        pass









