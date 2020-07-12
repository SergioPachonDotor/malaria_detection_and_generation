#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



from Data_Ind_cells import data_preparation
from Data_thick import data_creation
from Train import training
from Test import plot_AUC_ROC, evaluation_metrics
from model import CNN_model
import os
import glob
import sys

BATCH_SIZE = 64
EPOCHS = 25

##### train model with the individual cells datasets
def ind_cells_cls(base_dir, IMG_DIMS, INPUT_SHAPE ):
	train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = data_preparation(base_dir, IMG_DIMS)
	model = CNN_model(INPUT_SHAPE)
	model_trained = training(model, train_imgs, train_labels, val_imgs, val_labels, BATCH_SIZE, EPOCHS, model_name='CNN_ind_cells')
	return model_trained, test_imgs, test_labels
	

##### train model with the thick smear images datasets
def thick_img_cls(dir_, IMG_DIMS, INPUT_SHAPE):					                             
	train_imgs, train_labels, test_imgs, test_labels, val_imgs, val_labels = data_creation(dir_, IMG_DIMS, w=40, h=40, num_neg=20) 
	model = CNN_model(INPUT_SHAPE)
	model_trained = training(model, train_imgs, train_labels, val_imgs, val_labels, BATCH_SIZE, EPOCHS, model_name='CNN_thick_imgs')
	return model_trained, test_imgs, test_labels

if __name__ == '__main__':
	directory = sys.argv[1]
	if directory == './cell_images' or directory == 'cell_images':
		print('you choose to train individual cells')
		base_dir = os.path.join(directory)
		CNN_ind_cells, test_imgs, test_labels = ind_cells_cls(base_dir, IMG_DIMS=(100,100), INPUT_SHAPE=(100, 100, 3))
		print('####### Test #######')
		plot_AUC_ROC(CNN_ind_cells, test_imgs, test_labels, model_name='CNN_ind_cells')
		evaluation_metrics(CNN_ind_cells, test_imgs, test_labels)

	elif directory == './plasmodium-phonecamera' or directory == 'plasmodium-phonecamera':
		print('you choose to train thick smear images')
		dir_ = os.path.join(directory)
		CNN_thick_imgs, test_imgs, test_labels = thick_img_cls(dir_, IMG_DIMS=(50,50), INPUT_SHAPE=(50, 50, 3))
		print('####### Test #######')
		plot_AUC_ROC(CNN_thick_imgs, test_imgs, test_labels,model_name='CNN_thick_imgs')
		evaluation_metrics(CNN_thick_imgs, test_imgs, test_labels)

