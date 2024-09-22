#!/usr/bin/python
import os
import sys
import Image
import math
import matplotlib.image as mpimg
import numpy as np

label_file = 'dummy_submission.csv'

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
nc = 3

"""
Convert an array of binary labels to a uint8
Converts a binary image (array of 0s and 1s) into an 8-bit unsigned 
integer format (values scaled to 0 and 255, which are standard 
grayscale values for black and white pixels).
"""
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg

"""
Reconstructs the predicted image from the label data stored in the CSV.
"""
def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8) #Initialise empty image
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        # For each prediction, it extracts the coordinates of the 
        # patch from the filename format (image_id_x_y), then fills 
        # in the corresponding area in the im array with the prediction.
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png') #Reconstructed image

    return im

for i in range(1, 5):
    reconstruct_from_labels(i)

