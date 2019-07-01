import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
import sys
from moviepy.editor import VideoFileClip
import os

# user defined imports
from undistort import undistortImage
from thresholdBinaryImage import *
from showImageSideBySide import *
from applyPerspectiveTransform import *
from fitPolynomial import *


showOnlyFinalImage = True
video = True

# testimage = mpimg.imread('../test_images/straight_lines1.jpg')
def advanced_lane_detection_pipeline(testimage):
	global showOnlyFinalImage
	global video
	interMediateImagePlot = False
	# First correct image for distortion
	correctedImage = undistortImage(testimage)
	if interMediateImagePlot == True:
		showImageForComparison(testimage, correctedImage, "Original Image", "undistort Image", gray_new_img=False, text=None)

	# Then Apply perspective transformation to undistorted image
	warpedImage = getWarpedImage(correctedImage)
	if interMediateImagePlot == True:
		showImageForComparison(testimage, warpedImage, "Original Image", "Warped Image", gray_new_img=False, text=None)

	# Then apply binary threshold using sobel filter, color selection
	binary_image = getThresholdBinaryImage(warpedImage)
	if interMediateImagePlot == True:
		showImageForComparison(testimage, binary_image, "Original Image", "Binary Image", gray_new_img=True, text=None)

	final_image, left_lane_radius_m, right_lane_radius_m, vehicle_offset_m = fit_polynomial(binary_image, testimage, correctedImage, showOnlyFinalImage)

	# write text on final image. Choose any lane for radius as both are same
	if vehicle_offset_m < 0:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset to left: "+str(abs(vehicle_offset_m))+"m"
	elif vehicle_offset_m > 0:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset to right: "+str(vehicle_offset_m)+"m"
	else:
		text = "Radius: "+str(left_lane_radius_m)+"m" + "\noffset is at center: "+str(vehicle_offset_m)+"m"

	# if (showOnlyFinalImage == False) and (video == False):
	# 	plt.text(400, 100, text, fontsize=12, color='white')
	# 	plt.imshow(final_image)
	# 	plt.show()

	if (video == True):
		text = "Radius: "+str(left_lane_radius_m)+"m"
		cv2.putText(final_image, text, (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
		if vehicle_offset_m < 0:
			text = "offset to left: "+str(abs(vehicle_offset_m))+"m"
		elif vehicle_offset_m > 0:
			text = "offset to right: "+str(vehicle_offset_m)+"m"
		else:
			text = "offset is at center: "+str(vehicle_offset_m)+"m"
		cv2.putText(final_image, text, (200,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
		return final_image
	else:
		return final_image, text

def extract_frames(images):
	return images


def main():
	global showOnlyFinalImage
	global video
	# print command line arguments
	arg  = sys.argv[1:][0] # only supporting first arg
	if arg == "images":
		print("Advanced Lane detection on Images")
		video = False
		# laneimages = glob.glob('../video_debug_image/videoImagePrblm*.jpg')
		laneimages = glob.glob('../test_images/*.jpg')
		for img in laneimages:
			# print(img)
			# img = "../video_debug_image/videoImageOther14.jpg"
			# img = "../video_debug_image/videoImage7.jpg"
			# img = "../video_debug_image/videoImageLast4.jpg"
			# img = "../video_debug_image/videoImageLast0.jpg"
			# img ="../video_debug_image/videoImageLast10.jpg"
			# img = "../video_debug_image/videoImagePrblm4.jpg"
			image = mpimg.imread(img)
			pipeline_output_image, text = advanced_lane_detection_pipeline(image)
			# showImageForComparison(image, pipeline_output_image, "Original Image", "Final Image", gray_new_img=False, text=text)
			outputPath = "../output_images/"+img.split('/')[2]
			cv2.putText(pipeline_output_image, text, (200,140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
			mpimg.imsave(outputPath, pipeline_output_image)
			# break
	elif arg == "video":
		print("Advanced Lane detection on Video")
		video = True
		clip = VideoFileClip("../project_video.mp4")#.subclip(39.8, 40.2) # 1- 1.5 sec & 39 - 39.6 & 41.3 - 42, 39.8 - 40.2
		#Debugging as video between 39sec and 39.6 sec giving weird lane
		# clip.write_images_sequence("../video_debug_image/videoImagePrblm%01d.jpg")
		white_clip = clip.fl_image(advanced_lane_detection_pipeline)
		white_clip.write_videofile("../myprojectOutput.mp4", audio=False)

	else:
		print("Advanced Lane detection on Video")


if __name__ == '__main__':
	main()