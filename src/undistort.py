import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

# Flag to control undist and dist image show
showUndisImage = False

def undistortImage(image):
	cameraMtx = np.load('cameraMatrix.npy')
	cameraDistCoeff = np.load('distortionCoeff.npy')

	# now test whether distortion works or not
	undist = cv2.undistort(image, cameraMtx, cameraDistCoeff, None, cameraMtx)
	# From quiz to show original test image and undistorted image together
	if showUndisImage == True:
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(image)
		ax1.set_title('Original Image', fontsize=50)
		ax2.imshow(undist)
		ax2.set_title('Undistorted Image', fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()

	# return undistorted image
	return undist