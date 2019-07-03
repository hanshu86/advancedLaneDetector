import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def getThresholdBinaryImageHLS(img, s_thresh=(170, 255), sobelx_thresh=(50, 200), dir_thresh=(0, np.pi/8), sobel_kernel=5):
    mag_thresh=(50, 255)
    disablePlot = True
    # Convert to HLS color space and separate the V channel
    #image MUST be open using mpimg.imread(). If used cv2.imread
    # then change cv2.COLOR_RGB2HLS -> cv2.COLOR_BGR2HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # store different channel. We will use S & L -CHANNEL as it gives better image
  	# for lane detection
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    if disablePlot == False:
        plt.imshow(l_channel)
        plt.text(100,200, "L - Channel", fontsize=12, color='white')
       	plt.show()
        plt.imshow(s_channel)
        plt.text(100,200, "S - Channel", fontsize=12, color='white')
        plt.show()
        plt.imshow(h_channel)
        plt.text(100,200, "H - Channel", fontsize=12, color='white')
        plt.show()
    # Gradient on X direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize= sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Gradient on Y Dir
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize= sobel_kernel)
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # magnitue of sobel gradient in X & Y Diredction
    sobel_abs_mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor_mag = np.max(sobel_abs_mag)/255 
    scaled_sobel_mag = (sobel_abs_mag/scale_factor_mag).astype(np.uint8) 

    # plt.imshow(scaled_sobel_mag)
    # plt.show()

    # find direction of gradient
    dirGrad = np.arctan2(abs_sobely, abs_sobelx)
    if disablePlot == False:
        plt.imshow(dirGrad)
        plt.text(100,200, "Grad DIR", fontsize=12, color='white')
        plt.show()

    # starts applying threshold 
    binary_output_dir = np.zeros_like(sobelx)
    binary_output_dir[(dirGrad > dir_thresh[0]) & (dirGrad <= dir_thresh[1])] = 1

    if disablePlot == False:
        plt.imshow(binary_output_dir)
        plt.text(100,200, "binary on Grad DIR", fontsize=12, color='white')
        plt.show()

    # threshold for sobel magnitude of sobel gradient in X & Y direction
    binary_output_mag = np.zeros_like(scaled_sobel_mag)
    binary_output_mag[(scaled_sobel_mag >= mag_thresh[0]) & (scaled_sobel_mag <= mag_thresh[1])] = 1
    
    if disablePlot == False:
        plt.imshow(binary_output_mag)
        plt.text(100,200, "binary on Mag", fontsize=12, color='white')
        plt.show()

    # Threshold for x gradient on L-channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1
    
    if disablePlot == False:
        plt.imshow(sxbinary)
        plt.text(100,200, "binary on Grad X", fontsize=12, color='white')
        plt.show()

    # Threshold for Y gradient on L-channel (same threshold as X)
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= sobelx_thresh[0]) & (scaled_sobel_y <= sobelx_thresh[1])] = 1
    
    if disablePlot == False:
        plt.imshow(sybinary)
        plt.text(100,200, "binary on Grad Y", fontsize=12, color='white')
        plt.show()

    # Threshold color channel - S
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    if disablePlot == False:
        plt.imshow(s_binary)
        plt.text(100,200, "binary on S channel", fontsize=12, color='white')
        plt.show()

	# Threshold color channel - L
    sl_binary = np.zeros_like(l_channel)
    sl_binary[(l_channel >= 100) & (l_channel <= s_thresh[1])] = 1
    
    if disablePlot == False:
        plt.imshow(sl_binary)
        plt.text(100,200, "binary on L channel", fontsize=12, color='white')
        plt.show()

    color_binary = np.zeros_like(binary_output_mag)
    # color_binary[ ((sxbinary == 1) & (binary_dir_output == 1)) & ((s_binary == 1) | (sl_binary == 1)) ] = 1

    color_binary[ ((sxbinary == 1) & (sybinary == 1)) |  ((binary_output_mag == 1) & (binary_output_dir == 1)) | ( (sl_binary == 1) & (s_binary == 1)) ] = 1
    color_binary = color_binary #* 255
    # plt.imshow(color_binary)
    # plt.text(100,200, "Combined", fontsize=12, color='white')
    # plt.show()
    # Stack each channel
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary



def getThresholdBinaryImageLUVLAB(img, sobel_kernel=5, disablePlot=True):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    u_channel = luv[:,:,1]
    v_channel = luv[:,:,2]

    if disablePlot == False:
        plt.imshow(l_channel)
        plt.text(100,200, "LUV-L - Channel", fontsize=12, color='white')
        plt.show()
        plt.imshow(u_channel)
        plt.text(100,200, "U - Channel", fontsize=12, color='white')
        plt.show()
        plt.imshow(v_channel)
        plt.text(100,200, "V - Channel", fontsize=12, color='white')
        plt.show()

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab_channel = lab[:,:,0]
    a_lab_channel = lab[:,:,1]
    b_lab_channel = lab[:,:,2]

    if disablePlot == False:
        plt.imshow(l_lab_channel)
        plt.text(100,200, "L-LAB - Channel", fontsize=12, color='white')
        plt.show()
        plt.imshow(a_lab_channel)
        plt.text(100,200, "A - Channel", fontsize=12, color='white')
        plt.show()
        plt.imshow(b_lab_channel)
        plt.text(100,200, "B - Channel", fontsize=12, color='white')
        plt.show()

    # Gradient on L channel of LAB color space X direction
    sobelx_lab = cv2.Sobel(l_lab_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_lab = np.absolute(sobelx_lab) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_l_lab = np.uint8(255*abs_sobelx_lab/np.max(abs_sobelx_lab))

    # Gradient on B channel of LAB color space X direction
    sobelx_b_lab = cv2.Sobel(b_lab_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx_b_lab = np.absolute(sobelx_b_lab) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_b_lab = np.uint8(255*abs_sobelx_b_lab/np.max(abs_sobelx_b_lab))

    # for white line
    l_luv_binary = np.zeros_like(l_channel)
    l_luv_binary[(l_channel >= 190) & (l_channel <= 255)] = 1

    if disablePlot == False:
        plt.imshow(l_luv_binary)
        plt.text(100,200, "L-LUV - Binary", fontsize=12, color='white')
        plt.show()

    l_lab_binary = np.zeros_like(l_lab_channel)
    l_lab_binary[(l_lab_channel >= 190) & (l_lab_channel <= 255)] = 1

    if disablePlot == False:
        plt.imshow(l_lab_binary)
        plt.text(100,200, "L-LAB - Binary", fontsize=12, color='white')
        plt.show()

    sxbinary_l_lab = np.zeros_like(scaled_sobel_l_lab)
    sxbinary_l_lab[(scaled_sobel_l_lab >= 50) & (scaled_sobel_l_lab <= 200)] = 1

    if disablePlot == False:
        plt.imshow(sxbinary_l_lab)
        plt.text(100,200, "L-LAB Grad - Binary", fontsize=12, color='white')
        plt.show()

    # For Yellow line
    v_channel_binary = np.zeros_like(v_channel)
    v_channel_binary[(v_channel >= 170) & (v_channel <= 255)] = 1

    if disablePlot == False:
        plt.imshow(v_channel_binary)
        plt.text(100,200, "V-LUV - Binary", fontsize=12, color='white')
        plt.show()

    b_channel_binary = np.zeros_like(b_lab_channel)
    b_channel_binary[(b_lab_channel >= 155) & (b_lab_channel <= 255)] = 1

    if disablePlot == False:
        plt.imshow(b_channel_binary)
        plt.text(100,200, "B-LAB - Binary", fontsize=12, color='white')
        plt.show()

    sxbinary_b_lab = np.zeros_like(scaled_sobel_b_lab)
    sxbinary_b_lab[(scaled_sobel_b_lab >= 50) & (scaled_sobel_b_lab <= 200)] = 1

    if disablePlot == False:
        plt.imshow(sxbinary_b_lab)
        plt.text(100,200, "B-LAB Grad - Binary", fontsize=12, color='white')
        plt.show()

    combined_binary = np.zeros_like(b_channel_binary)

    # combined_binary[ ((l_luv_binary == 1) & (l_lab_binary == 1)) | 
    #                  ((v_channel_binary == 1) & (b_channel_binary == 1) ) ] = 1

    combined_binary[ ((l_luv_binary == 1) & (l_lab_binary == 1)) | 
                     (sxbinary_l_lab == 1) | 
                     ((v_channel_binary == 1) & (b_channel_binary == 1)) |
                     (sxbinary_b_lab == 1) ] = 1

    if disablePlot == False:
        plt.imshow(combined_binary)
        plt.text(100,200, "Luv/Lab - Binary", fontsize=12, color='white')
        plt.show()

    return combined_binary





