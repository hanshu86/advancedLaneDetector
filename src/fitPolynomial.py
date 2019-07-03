import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
import scipy

# user defined imports
from undistort import undistortImage
from thresholdBinaryImage import *
from showImageSideBySide import *
from applyPerspectiveTransform import *
from drawOnLane import *

# return radius of the curvature
def get_curvead(A, B, y):
    part1 = np.square((2 * A * y) + B)
    part2 = (1 + part1)
    part3 = part2*part2*part2
    part4 = np.sqrt(part3)
    part5 = np.abs(2*A)
    return part4/part5;

def getHistogram(img):
	plotHistoGram = False
	# Take a histogram of the bottom half of the image
	bottom_half = img[img.shape[0]//4:,:]
	histogram = np.sum(bottom_half, axis=0)
	if plotHistoGram == True:
		plt.plot(histogram)
		plt.show()
	return histogram

def windowStartPoint(hist):
	midpoint = np.int(hist.shape[0]//2)
	leftx_start = np.argmax(hist[:midpoint])
	rightx_start = np.argmax(hist[midpoint:]) + midpoint
	return leftx_start, rightx_start

# From Quiz
def find_lane_pixels(binary_warped):
	histogram = getHistogram(binary_warped)
	# MUST CONVERT DTYPE TO UINT8 as DTYPE here is float64
	binary_warped = binary_warped.astype('uint8')
	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	leftx_base,rightx_base = windowStartPoint(histogram)

	# HYPERPARAMETERS
	# Choose the number of sliding windows
	nwindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated later for each window in nwindows
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
	    # Identify window boundaries in x and y (and right and left)
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    # Find the four below boundaries of the window ###
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin

	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
	    	    
	    # Append these indices to the lists
	    left_lane_inds.append(good_left_inds)
	    right_lane_inds.append(good_right_inds)
	    
	    ### If found > minpix pixels, recenter next window ###
	    ### (`right` or `leftx_current`) on their mean position ###
	    if len(good_left_inds) >= minpix:
	        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	        
	    if len(good_right_inds) >= minpix:
	        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of lists of pixels)
	try:
	    left_lane_inds = np.concatenate(left_lane_inds)
	    right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
	    # Avoids an error if the above is not implemented fully
	    pass

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	return leftx, lefty, rightx, righty, out_img


def is_distance_good(left_fit, right_fit, binary_warped):
	isGood = True
	smally = np.linspace(0, binary_warped.shape[0]-1, 3 )
	#print(smally)
	x_left = left_fit[0]*smally**2 + left_fit[1]*smally + left_fit[2]
	x_right = right_fit[0]*smally**2 + right_fit[1]*smally + right_fit[2]
	dist = (x_right - x_left)
	#print(dist)
	for i in range(0, len(dist)):
		if dist[i] < 800 or dist[i] > 1100:
			isGood = False
			break
	return isGood

# From Quiz
def fit_polynomial(binary_warped, org_image, undist_image, showOnlyFinalImage):
	#showOnlyFinalImage = False
	# Find our lane pixels first
	leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

	# Fit a second order polynomial to each using `np.polyfit` ###
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	isLaneProperlySeparated = is_distance_good(left_fit, right_fit, binary_warped)
	if isLaneProperlySeparated == False:
		# print("In this Frame Lane lines are NOT properly separated.. Use lane lines from previous Frame")
		left_fit = np.load('leftFit.npy')
		right_fit = np.load('rightFit.npy')
	else:
		np.save('leftFit.npy', left_fit)
		np.save('rightFit.npy', right_fit)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

	try:
	    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	except TypeError:
	    # Avoids an error if `left` and `right_fit` are still none or incorrect
	    print('The function failed to fit a line!')
	    left_fitx = 1*ploty**2 + 1*ploty
	    right_fitx = 1*ploty**2 + 1*ploty

	## Visualization ##
	# Colors in the left and right lane regions
	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]

	# Plots the left and right polynomials on the lane lines
	if showOnlyFinalImage == False:
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.imshow(out_img)
		plt.show()

	# assume the lane is about 30 meters long and 3.7 meters wide
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 12/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/980 # meters per pixel in x dimension (1280 org size. perspective transform --->100 (980) <---200)
	y_eval = np.max(ploty) # bottom of the image
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

	left_curve_rad = get_curvead(left_fit_cr[0], left_fit_cr[1], y_eval*ym_per_pix)  ## left lane radius
	right_curve_rad = get_curvead(right_fit_cr[0], right_fit_cr[1], y_eval*ym_per_pix)  ## right lane radius

	# Now get the position of the vehicle
	camera_pos = binary_warped.shape[1]/2
	left_xfit_pos =  left_fit[0] * binary_warped.shape[0]**2 + left_fit[1] * binary_warped.shape[0] + left_fit[2]
	right_xfit_pos =  right_fit[0] * binary_warped.shape[0]**2 + right_fit[1] * binary_warped.shape[0] + right_fit[2]
	lane_middle_pos = (left_xfit_pos + right_xfit_pos)/2
	car_offset = (camera_pos - lane_middle_pos) * xm_per_pix

	final_img = drawOnLane(binary_warped, left_fitx, right_fitx, ploty, org_image, undist_image)

	return final_img, left_curve_rad, right_curve_rad, car_offset

