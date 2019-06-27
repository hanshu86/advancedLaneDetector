import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getThresholdBinaryImage(img, s_thresh=(170, 255), sobelx_thresh=(20, 100), dir_thresh=(0, np.pi/2), sobel_kernel=5):
    # Convert to HLS color space and separate the V channel
    #image MUST be open using mpimg.imread(). If used cv2.imread
    # then change cv2.COLOR_RGB2HLS -> cv2.COLOR_BGR2HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # store different channel. We will use S-CHANNEL as it gives better image
  	# for lane detection
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x on S-channel
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize= sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Gradient direction
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize= sobel_kernel)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # Threshold gradient direction
    binary_dir_output = np.zeros_like(sobelx)
    binary_dir_output[(grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.zeros_like(binary_dir_output)
    color_binary[ ((sxbinary == 1) & (binary_dir_output == 1)) | (s_binary == 1) ] = 1
    # Stack each channel
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary