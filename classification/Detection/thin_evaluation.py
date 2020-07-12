#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 
from skimage import measure
import sys
import time
import numpy as np

def fillholes(img):
	im = img.copy()
	h,w = im.shape[:2]
	
	## the mask of the fill should be greater than the image
	mask = np.zeros((h+2, w+2), np.uint8)
	seedpoint = (0, 0) # starting point 
	## if the first pixel is not a background, check the closest background to be the starting point 
	if im[0,0] != 0:
		for i in range(h):
			for j in range(w):
				if im[i,j] == 0:
					seedpoint = (i,j)
					break
	## filling the holes
	_,im_fill,_,_ = cv2.floodFill(im, mask, seedpoint,255)
	im_fill_inv = cv2.bitwise_not(im_fill)
	## combine the image filled with the original
	img_fill = img | im_fill_inv
	return img_fill

def segmentation_cells(image_name):
	try:
		image = cv2.imread(image_name)
	except:
		print('Image not found')
	im_copy = image.copy()
    
	##### cells segmentation #########
	gray_im = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)

	## increase the contrast of the image
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
	cl = clahe.apply(gray_im)

	#dividing the region in black and white
	ret, thresh = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	#removing the noise
	#morphology consists of dilating or eroding the white region
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 3)

	open_new = fillholes(opening) #filling the holes

	# #find the sure bagckround
	sure_bg = cv2.dilate(open_new, kernel, iterations = 5)


	# #find the sure foreground
	dist_transform = cv2.distanceTransform(open_new, cv2.DIST_L2, maskSize=5)
	dist_transform =  np.uint8(dist_transform)
	## the separation of the cells depend on this threshold, it will depend on the image, 
	##testing on different images, I found that 9 gave the best separation  
	ret, sure_fg = cv2.threshold(dist_transform, 9, 255, cv2.THRESH_BINARY)

	sure_fg = cv2.erode(sure_fg, kernel)

	# #find the unkown region
	unknown = cv2.subtract(sure_bg, sure_fg)

	# #label the foreground objects
	ret, label = cv2.connectedComponents(sure_fg)


	# #add to all labels so that the sure_bg = 1 not 0
	label += 1

	# #label the unkown region as 0
	label[unknown == 255] = 0

	#find the contour of each cells
	markers = cv2.watershed(im_copy , label)

	######## taking the regions with respect to the markers ########
	regions = measure.regionprops(markers, intensity_image=cl)

	#put into red these contours
	im_copy[markers == -1] = [0, 255, 255]
	cv2.imwrite(image_name+'seg.png', im_copy)
	plt.figure(figsize=(10,10))
	plt.imshow(cv2.cvtColor(im_copy, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()
	return regions,image
	

    
    ####### cells classifications #########
def classification_cells(image_name, model):
	t = time.time()
	count_parasites = 0
	count_cells = 0
	regions, image = segmentation_cells(image_name)
	image_cp = image.copy()
	
	for prop in regions[1:]: #we took off the first region because it seems it's for the border
		count_cells += 1
		#take only the region containing one cell, accodring to their area
		if prop.area < 2000:
			y_min, x_min, y_max, x_max = prop.bbox ##### taking a box for each regions
			#take a patche of the image according to the bounding box
			cells = image[y_min:y_max, x_min:x_max]
			cell = cv2.resize(cells,(100,100))/255.
            
			cell = tf.expand_dims(cell, 0)

			predict = model.predict(cell)     #### predict the cell with our model
			predict = np.where(predict[0][0]>0.5,1,0)

			if int(predict) == 1:
				cy,cx = prop.centroid
				cv2.circle(image_cp, (int(cx), int(cy)), int((x_max-x_min)/2), (0,0,255), 1) ### put a circle in the infected cells
				count_parasites += 1
      
	elapsed_time = time.time() - t
	print('Time for segmentation and detection: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
	cv2.imwrite(image_name+'cls.png', image_cp)
	print('Number of cells segmented {}'.format(count_cells))
	print('Number of infected cells {}'.format(count_parasites))
    
	plt.figure(figsize=(10,10))
	plt.imshow(cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()
   

if __name__ == '__main__':
	model_dir = sys.argv[1]
	image_name = sys.argv[2]
	try:
		model = tf.keras.models.load_model(model_dir)
		model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
		print('Model loaded successfully')
	except:
		print('Model not found')
	classification_cells(image_name=image_name, model=model)












