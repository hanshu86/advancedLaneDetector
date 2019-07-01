import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob


def doCameraCalibration():
	# Read calibration image
	images = glob.glob('../camera_cal/calibration*.jpg')

	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# images has 9x6 corners
	objPoints = [] # to hold 3D points in real world space
	imgPoints = [] # to hold 2D points in image space

	nx = 9
	ny = 6
	# prepare object points
	objp = np.zeros((ny*nx, 3), np.float32)
	# leave Z coordinate as zero
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
	global shape

	for calImage in images:
		image = mpimg.imread(calImage)
		# convert the image to grayscale for finding corners
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		shape = gray.shape[::-1]
		# plt.imshow(gray, cmap='gray')
		# plt.show()

		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

		if ret == True:
			# it found corners in 17 images out of 20
			#corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgPoints.append(corners)
			objPoints.append(objp) # this will be same for all image as this is in real world space

			img = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
			# plt.imshow(img)
			# plt.show()

	# here we have image points and object points which we can use to calibrate camera
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, shape, None, None)

	# save camera matrix and distortion coefficient for further use
	np.save('cameraMatrix.npy', mtx)
	np.save('distortionCoeff.npy', dist)

	return mtx, dist

# Uncomment below lines to verify camera calibration and image distortion correction
# testimage = mpimg.imread('../test_images/test1.jpg')
# undistortImage(testimage)

