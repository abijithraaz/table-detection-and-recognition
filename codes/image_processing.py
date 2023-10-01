import os
import cv2
import numpy as np
from PIL import Image

# Some image process techniques to improve the images.
class ImageProcessor():
    def __init__(self):
        pass

    def PIL_to_cv2(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 

    def cv2_to_PIL(self, cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    def image_padding(self, image, padd):
        '''
        Image boarder padding to avoid table image loss
        '''
        width, height = image.size
        new_width = width +(2*padd)
        new_height = height + (2*padd)
        color = (255, 255, 255)
        result = Image.new(image.mode, (new_width, new_height), color)
        result.paste(image, (padd, padd))
        return result


    def sharpen_image(self, pil_img):
        img = self.PIL_to_cv2(pil_img)
        '''
        Image sharpening kernal
        '''
        sharpen_kernel = np.array([[-1, -1, -1], 
                                [-1,  9, -1], 
                                [-1, -1, -1]])

        sharpen = cv2.filter2D(img, -1, sharpen_kernel)
        pil_img = self.cv2_to_PIL(sharpen)
        return pil_img

    def binarizeBlur_image(self, pil_img):
        image = self.PIL_to_cv2(pil_img)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)[1]

        result = cv2.GaussianBlur(thresh, (3,3), 0)
        result = 255 - result
        return self.cv2_to_PIL(result)
    
    def whole_image_processing(self, pil_img):
        sharpen_img = self.sharpen_image(pil_img)
        binary_img = self.binarizeBlur_image(sharpen_img)

        return binary_img
