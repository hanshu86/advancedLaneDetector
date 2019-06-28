import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob

# user defined imports
from undistort import undistortImage
from thresholdBinaryImage import *
from showImageSideBySide import *
from applyPerspectiveTransform import *

testimage = mpimg.imread('../test_images/straight_lines2.jpg')
correctedImage = undistortImage(testimage)
# showImageForComparison(testimage, correctedImage, "Original Image", "undistort Image")
warped = getWarpedImage(correctedImage)
showImageForComparison(testimage, warped, "Original Image", "Warped Image")
#binary_image = getThresholdBinaryImage(correctedImage)
#showImageForComparison(testimage, binary_image, "Original Image", "Binary Image", gray_new_img=True)