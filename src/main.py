import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
import sys

# user defined imports
from undistort import undistortImage
from thresholdBinaryImage import *
from showImageSideBySide import *
from applyPerspectiveTransform import *
from fitPolynomial import *


showOnlyFinalImage = True

# testimage = mpimg.imread('../test_images/straight_lines1.jpg')
def advanced_lane_detection_pipeline(testimage):
	# First correct image for distortion
	correctedImage = undistortImage(testimage)
	# showImageForComparison(testimage, correctedImage, "Original Image", "undistort Image")

	# Then Apply perspective transformation to undistorted image
	warpedImage = getWarpedImage(correctedImage)
	# showImageForComparison(testimage, warpedImage, "Original Image", "Warped Image")

	# Then apply binary threshold using sobel filter, color selection
	binary_image = getThresholdBinaryImage(warpedImage)
	#showImageForComparison(testimage, binary_image, "Original Image", "Binary Image", gray_new_img=True)

	final_image, left_lane_radius_m, right_lane_radius_m, vehicle_offset_m = fit_polynomial(binary_image, testimage, correctedImage, showOnlyFinalImage)

	# write text on final image. Choose any lane for radius as both are same
	if vehicle_offset_m < 0:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset to left: "+str(abs(vehicle_offset_m))+"m"
	elif vehicle_offset_m > 0:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset to right: "+str(vehicle_offset_m)+"m"
	else:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset is at center: "+str(vehicle_offset_m)+"m"

	if showOnlyFinalImage == False:
		plt.text(400, 100, text, fontsize=12, color='white')
		plt.imshow(final_image)
		plt.show()

	return final_image, text



def main():
	# print command line arguments
	arg  = sys.argv[1:][0] # only supporting first arg

	if arg == "images":
		print("Advanced Lane detection on Images")
		laneimages = glob.glob('../test_images/*.jpg')
		for img in laneimages:
			image = mpimg.imread(img)
			pipeline_output_image, text = advanced_lane_detection_pipeline(image)
			showImageForComparison(image, pipeline_output_image, "Original Image", "Final Image", gray_new_img=False, text=text)
			break
	elif arg == "video":
		print("Advanced Lane detection on Video")
	else:
		print("Advanced Lane detection on Video")

if __name__ == '__main__':
	main()