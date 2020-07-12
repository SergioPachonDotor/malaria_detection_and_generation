#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################



import tensorflow as tf
import datetime 
import time
import os
import numpy as np

import matplotlib.pyplot as plt

############## PLOT ACCURACY #########
def plot_accuracy(history, model_name):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
	t = f.suptitle('Basic CNN Performance', fontsize=12)
	f.subplots_adjust(top=0.85, wspace=0.3)

	max_epoch = len(history.history['accuracy'])+1
	epoch_list = list(range(1,max_epoch))
	ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
	ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
	ax1.set_xticks(np.arange(1, max_epoch, 5))
	ax1.set_ylabel('Accuracy Value')
	ax1.set_xlabel('Epoch')
	ax1.set_title('Accuracy')
	l1 = ax1.legend(loc="best")

	ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
	ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
	ax2.set_xticks(np.arange(1, max_epoch, 5))
	ax2.set_ylabel('Loss Value')
	ax2.set_xlabel('Epoch')
	ax2.set_title('Loss')
	l2 = ax2.legend(loc="best")
	plt.savefig('figures/'+model_name+'_Accuracy.png')
	plt.show()
	

############ TRAINING THE MODEL ##################
def training(model, train_imgs, train_labels, val_imgs, val_labels, BATCH_SIZE, EPOCHS, model_name):
	start_time = time.time()
	logdir = os.path.join('./callbacks/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              					patience=2, min_lr=0.000001)

                                             
	callbacks = [reduce_lr, tensorboard_callback]

	history = model.fit(x=train_imgs, y=train_labels, 
                    		batch_size=BATCH_SIZE,
                    		epochs=EPOCHS, 
                    		validation_data=(val_imgs, val_labels), 
                    		callbacks=callbacks,
                    		verbose=1)

	elapsed_time = time.time() - start_time
	print("Time of the training")
	print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
	
	plot_accuracy(history, model_name)
	model.save('models/'+model_name+'.h5')
	return model






