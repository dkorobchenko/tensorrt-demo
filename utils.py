'''
Utility functions
'''

import numpy as np
import scipy.misc

__author__ = "Dmitry Korobchenko (dkorobchenko@nvidia.com)"

def resize_and_crop(img_in, crop_size):
    '''
    Resize input image to fit the crop_size, keeping the aspect ratio
    Then crop
    -> img_in: input image
    -> crop_size: crop size
    <- resized and cropped image
    '''
    img = img_in.copy()

    if crop_size[1]/crop_size[0] < img.shape[1]/img.shape[0]:
        new_h = crop_size[0]
        new_w = int(np.ceil(crop_size[0] * img.shape[1] / img.shape[0]))
    else:
        new_h = int(np.ceil(crop_size[1] * img.shape[0] / img.shape[1]))
        new_w = crop_size[1]
    img = scipy.misc.imresize(img, (new_h, new_w), 'bilinear')

    offset = ((img.shape[0] - crop_size[0]) // 2,
              (img.shape[1] - crop_size[1]) // 2)
    img = img[offset[0]:offset[0]+crop_size[0],
              offset[1]:offset[1]+crop_size[1], :]

    return img