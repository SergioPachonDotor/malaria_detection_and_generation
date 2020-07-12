#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os 
import glob
from sklearn.model_selection import train_test_split


#taking the position of the object(plasmodium) in an image if there exist
def get_position_plasmodium(image_name):
	xml_file = image_name[:-3] + 'xml'
    
	tree = ET.parse(xml_file)
	root = tree.getroot()
    
	#if there is no object
	if len(root.findall('object')) == 0:
		return np.array([])
	else:
		pos_plasm = []
		for item in root.findall('object'):
			box = item.find('bndbox')
			xmin = round(float(box.find('xmin').text))
			xmax = round(float(box.find('xmax').text))
			ymin = round(float(box.find('ymin').text))
			ymax = round(float(box.find('ymax').text))

			xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
			pos_plasm.append((xmin, xmax, ymin, ymax))
		return np.vstack(pos_plasm)


####### Create a data from a pacthes of an image using the annotated data #################

# w, h are the width and height of the patches
# num_neg is the number maximal of the negative patches(without plasmodium)
def create_patches(dir_, w, h, num_neg):
	try:
		image_dir = glob.glob(dir_ + '/*.jpg')
	except:
		print('Directory not found')
	print('Total images: ',len(image_dir))
	count_no_object = 0
	count_object = 0

	#the output patches and label
	patches = []
	labels = []
	num_image = 0

	print('########## Data creation ##########')
	#getting the patches of the plasmodium and create a random negative patches
	for image in image_dir:
		num_image += 1
		if num_image % 500 == 0 or num_image == len(image_dir)-1:
			print('{} images done'.format(num_image))
    
   		#taking the object present in an image
		bbox = get_position_plasmodium(image_name=image)
		im = cv2.imread(image)
		width, height, c = im.shape
    
    		#taking the positive patchs
    
		if len(bbox) == 0:      #if no object 
			count_no_object += 1
		else:
			for box in bbox:     #for each plasmodium create a patch from the image
				x = np.where(box[0] < 0, 0, box[0])
				y = np.where(box[2] < 0, 0, box[2]) 
				patches.append(im[y:y+h, x:x+w])
				count_object += 1  
				labels.append(1)   #label corresponding to the presence of plasmodium
        
    
    		#taking the random negative patches
   
		for j in range(num_neg):  
			x = np.random.randint(low=0, high=width)
			y = np.random.randint(low=0, high=height)
			xmax = w + x
			ymax = h + y
        
        		#looking for overlap between positive and negative patch
        		#not consider the patche if there is a plasmodium inside
			overlap = False
			if len(bbox) == 0:
				pass
			else:
				for box in bbox:
                			#avoid overlap on the patches of positive and negative
					xmin, xmax, ymin, ymax = box
					xmin = np.where(xmin<0,0,xmin)
					ymin = np.where(ymin<0,0,ymin)
					cx = xmin + int((xmax - xmin)/2)
					cy = ymin + int((ymax - ymin)/2)
					if x<= cx <= xmax and y <= cy <= ymax:
						overlap = True

			if not overlap:
				patches.append(im[y:y+h, x:x+w])
				labels.append(0) #label corresponding to a negative patch
            
	print('Number of object {}'.format(count_object))
	print('Number of images without object {}'.format(count_no_object))
	return patches, labels


######## create an array from the patches, labels and split the data ################
def data_creation(dir_, IMG_DIMS, w, h, num_neg):
	patches, labels = create_patches(dir_, w, h, num_neg)
	labels = np.asarray(labels)
	#resize the image data and split in to train, validation and test
	train = np.zeros((len(patches),IMG_DIMS[0],IMG_DIMS[1],3))
	for i in range((len(patches))):
		train[i] = cv2.resize(patches[i],IMG_DIMS)

	print('The number of data {}'.format(len(train)))


	train_imgs, test_imgs, train_labels, test_labels = train_test_split(train,
                                                                      		labels,
                                                                      		test_size = 0.3, random_state = 42)
	train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_imgs,
                                                                  		train_labels,
                                                                  		test_size = 0.1, random_state = 42)
	print('Number of train images {}'.format(len(train_imgs)))
	print('Number of test images {}'.format(len(test_imgs)))
	print('Number of validation images {}'.format(len(val_imgs)))

	train_imgs = train_imgs/255.
	test_imgs = test_imgs/255.
	val_imgs = val_imgs/255.

	#to save the data
	#np.savez('train.npz', img=train_imgs, label =train_labels)
	#np.savez('val.npz', img=val_imgs, label=val_labels)
	#np.savez('test.npz', img= test_imgs, label=test_labels)

	return train_imgs, train_labels, test_imgs, test_labels, val_imgs, val_labels







