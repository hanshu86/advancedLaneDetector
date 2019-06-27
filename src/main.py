import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob

# user defined imports
from undistort import undistortImage
from thresholdBinaryImage import *
from showImageSideBySide import *


testimage = mpimg.imread('../test_images/test1.jpg')
correctedImage = undistortImage(testimage)
binary_image = getThresholdBinaryImage(correctedImage)
showImageForComparison(testimage, binary_image, "Original Image", "Binary Image", gray_new_img=True)