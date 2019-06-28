import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

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

	srcPoints = np.float32([ [256, 720], [610 , 430], [700, 430], [1100,  720] ])

	offset = 100
	img_size = (img.shape[1], img.shape[0])
	dstPoints = np.float32([ [offset, 720], [offset, 0], [ 300, 0], [img_size[1] - offset, 720]])

	persPectiveMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
	np.save('persPectiveMatrix.npy', persPectiveMatrix)
	return persPectiveMatrix

def getWarpedImage(img):
	try:
		warpedMatrix = np.load('persPectiveMatrix')
	except:
		print("Perspective Matrix Does not exists..create one")
		warpedMatrix = createSavePerspectiveMatrix(img)

	img_size = (img.shape[1], img.shape[0])
	print(img_size)
	warped = cv2.warpPerspective(img, warpedMatrix, img_size, flags=cv2.INTER_LINEAR)
	return warped
