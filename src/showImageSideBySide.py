import matplotlib.pyplot as plt 

def showImageForComparison(org_image, new_image, org_image_name, new_image_name, gray_new_img=False, text=None):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(org_image)
	ax1.set_title(org_image_name, fontsize=50)
	if gray_new_img == False:
		ax2.imshow(new_image)
	else:
		ax2.imshow(new_image, cmap='gray')
	ax2.set_title(new_image_name, fontsize=50)
	if len(text) > 0:
			ax2.text(400, 100, text, fontsize=12, color='white')
			ax2.imshow(new_image)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()
