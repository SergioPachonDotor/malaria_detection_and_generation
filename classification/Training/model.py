#####################################################
                                                    
# written by Rija Tonny C. Ramarolahy               
                                                     
# 2020                                               
                                                    
# email: rija@aims.edu.gh / minuramaro@gmail.com   
                                                     
#####################################################


import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

#################################### CNN MODEL ##########################
def CNN_model(INPUT_SHAPE):
	input = tf.keras.layers.Input(shape=INPUT_SHAPE)

	conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(input)
	pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
	pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
	pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

	flatten = tf.keras.layers.Flatten()(pool3)

	hidden1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
	drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
	hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
	drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

	output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

	model = tf.keras.Model(inputs=input, outputs=output)
	model.compile(optimizer='adam',
                	loss='binary_crossentropy',
                	metrics=['accuracy'])
	model.summary()

	return model
