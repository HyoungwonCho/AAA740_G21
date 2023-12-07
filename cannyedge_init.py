import os
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

image_path = '/media/dataset/init_hw/night_temp'
mask_only_path = '/media/dataset/init_hw/night_mask_only_new'
save_path = '/media/dataset/init_hw/night_mask_temp'

temp_path = '/media/dataset/init_hw/night_mask_only_temp'

for image_filename in os.listdir(image_path):
    
    image = cv2.imread(os.path.join(image_path, image_filename))
    mask_only = cv2.imread(os.path.join(mask_only_path, image_filename.replace('night', 'night_mask_')))
    
    cv2.imwrite(os.path.join(temp_path, image_filename.replace('night', 'night_mask')), mask_only)
    
    image = cv2.resize(image, (256, 256))
    mask_only = cv2.resize(mask_only, (256, 256))
    
    edges = cv2.Canny(image,90,180)/255
    thres = np.array((mask_only[:,:,0]<100) & (mask_only[:,:,1]>100))
    maksed_edges = 1 - thres*edges

    newmask = np.array([mask_only[:,:,0]*maksed_edges, mask_only[:,:,1]*maksed_edges, mask_only[:,:,2]*maksed_edges]).transpose((1,2,0))
    newmask = np.uint8(newmask)

    cv2.imwrite(os.path.join(save_path, image_filename.replace('night', 'night_mask')), newmask)