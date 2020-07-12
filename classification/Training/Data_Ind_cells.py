#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



import os
import glob
import pandas as pd
import numpy as np
#for splitting data
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from concurrent import futures
from sklearn.preprocessing import LabelEncoder
import threading


np.random.seed(42)

def resize_img(idx, img, IMG_DIMS, total_img):
	if idx % 5000 == 0 or idx == (total_img - 1):
        	print('{}: working on img num: {}'.format(threading.current_thread().name, idx))
	img = cv2.imread(img)
	img = cv2.resize(img, dsize = IMG_DIMS,
                     interpolation = cv2.INTER_CUBIC)
	img = np.array(img, dtype=np.float32)
	return img

def data_preparation(base_dir, IMG_DIMS):
	try:
		infected_dir = os.path.join(base_dir,'Parasitized')
		healthy_dir = os.path.join(base_dir,'Uninfected')
	except:
		print('Directory not found')

	#counting the number of the images in each files
	infected_files = glob.glob(infected_dir+'/*.png')
	healthy_files = glob.glob(healthy_dir+'/*.png')
	#print(len(infected_files), len(healthy_files))


	########### PREPARING DATA ##############

	#build a dataframe for the data
	files_df = pd.DataFrame({'filename': infected_files + healthy_files,
    				'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)}).sample(frac=1, random_state=42).reset_index(drop=True)
	#print(files_df.head())

	#split the data
	train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      		files_df['label'].values,
                                                                      		test_size = 0.3, random_state = 42)
	train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                   		train_labels,
                                                                   		test_size = 0.1, random_state = 42)
	print("training data: ", train_files.shape, " \n ", Counter(train_labels))
	print("validation data: ", val_files.shape, '\n ', Counter(val_labels))
	print("test data: ", test_files.shape, '\n ', Counter(test_labels))
	

	ex = futures.ThreadPoolExecutor(max_workers=None)
	train_data_inp = [(idx, img, IMG_DIMS, len(train_files)) for idx, img in enumerate(train_files)]
	val_data_inp = [(idx, img, IMG_DIMS, len(val_files)) for idx, img in enumerate(val_files)]
	test_data_inp = [(idx, img, IMG_DIMS, len(test_files)) for idx, img in enumerate(test_files)]
	
	#resize images in each data
	print('Loading Train Images:')
	train_data_map = ex.map(resize_img,
                        	[record[0] for record in train_data_inp],
                        	[record[1] for record in train_data_inp],
                       	 	[record[2] for record in train_data_inp],
				[record[3] for record in train_data_inp])
	train_data = np.array(list(train_data_map))

	print('\n Loading Validation Images:')
	val_data_map = ex.map(resize_img,
                        	[record[0] for record in val_data_inp],
                        	[record[1] for record in val_data_inp],
                        	[record[2] for record in val_data_inp],
				[record[3] for record in val_data_inp])
	val_data = np.array(list(val_data_map))

	print('\n Loading Test Images:')
	test_data_map = ex.map(resize_img,
                        	[record[0] for record in test_data_inp],
                        	[record[1] for record in test_data_inp],
                        	[record[2] for record in test_data_inp],
				[record[3] for record in test_data_inp])
	test_data = np.array(list(test_data_map))
 

	### normalize data and save it
	train_imgs = train_data/255.
	val_imgs = val_data/255.
	test_imgs = test_data/255.


	le = LabelEncoder()
	le.fit(train_labels)
	train_labels = le.transform(train_labels)
	val_labels = le.transform(val_labels)
	test_labels = le.transform(test_labels)
		
	#to save the data
	#np.savez('train.npz', img=train_imgs, label =train_labels)
	#np.savez('val.npz', img=val_imgs, label=val_labels)
	#np.savez('test.npz', img= test_imgs, label=test_labels)

	return train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels








