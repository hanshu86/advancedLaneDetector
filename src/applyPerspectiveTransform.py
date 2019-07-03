import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage

from undistort import undistortImage
from showImageSideBySide import *

# testimage = mpimg.imread('../examples/warped_straight_lines.jpg')
# plt.imshow(testimage)
# plt.show()
# exit()
# #print(testimage.shape)
# source image points

def createSavePerspectiveMatrix(img):
	# srcPoints = np.float32([  	[167, 600],  # left bottom
	#                     		[494 , 374], # left top
	#                     		[572, 374], # right top
	#                     		[931,  600] ]) # right bottom

	# dstPoints = np.float32([  	[257, 600],  # left bottom
	#                     		[257 , 0], # left top
	#                     		[786, 0], # right top
	#                     		[786,  600] ]) # right bottom

	srcPoints = np.float32([ [270, 683], [559 , 477], [734, 477], [1044,  675] ])

	offset = 100
	img_size = (img.shape[1], img.shape[0])
	dstPoints = np.float32([ [offset, 720], [offset, 0], [ img_size[0] - 2*offset, 0], [img_size[0] - 2*offset, 720]])

	print(dstPoints)
	persPectiveMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
	persPectiveMatrixInv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
	np.save('persPectiveMatrix.npy', persPectiveMatrix)
	np.save('persPectiveMatrixInv.npy', persPectiveMatrixInv)
	return persPectiveMatrix

def getWarpedImage(img):
	try:
		warpedMatrix = np.load('persPectiveMatrix.npy')
	except:
		print("Perspective Matrix Does not exists..create one")
		warpedMatrix = createSavePerspectiveMatrix(img)

	img_size = (img.shape[1], img.shape[0])
	#print(img_size)
	warped = cv2.warpPerspective(img, warpedMatrix, img_size, flags=cv2.INTER_LINEAR)
	return warped


# PLEASE UNCOMMENT ME FOR RUNNING THIS FILE ONLY
# distortedImage = mpimage.imread('../test_images/straight_lines2.jpg')
# imageCorr = undistortImage(distortedImage)
# warpedImg = getWarpedImage(imageCorr)
# showImageForComparison(distortedImage, warpedImg, "Original Image", "warped Image", gray_new_img=False, text=None)
# plt.imshow(warpedImg)
# plt.show()