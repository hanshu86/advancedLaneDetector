import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

#testimage = mpimg.imread('../examples/warped_straight_lines.jpg')
#print(testimage.shape)
# source image points
srcPoints = np.float32([  	[167, 600],  # left bottom
                    		[494 , 374], # left top
                    		[572, 374], # right top
                    		[931,  600] ]) # right bottom

dstPoints = np.float32([  	[257, 600],  # left bottom
                    		[257 , 0], # left top
                    		[786, 0], # right top
                    		[786,  600] ]) # right bottom


persPectiveMatrix = cv2.getPerspectiveTransform(src, dst)

img_size = (gray.shape[1], gray.shape[0])
warped = cv2.warpPerspective(undist, persPectiveMatrix, img_size, flags=cv2.INTER_LINEAR)
