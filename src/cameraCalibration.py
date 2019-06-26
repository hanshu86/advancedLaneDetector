import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob

# Read calibration image
images = glob.glob('../camera_cal/calibration*.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# images has 9x6 corners
objPoints = [] # to hold 3D points in real world space
imgPoints = [] # to hold 2D points in image space

ny = 6
nx = 9
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

# now test whether distortion works or not
testimage = mpimg.imread('../camera_cal/test_image.png')
undist = cv2.undistort(testimage, mtx, dist, None, mtx)
# From quiz to show original test image and undistorted image together
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(testimage)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()