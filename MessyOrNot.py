# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:39:40 2019

@author: 138410
"""


import tkinter as tk
from tkinter import filedialog
#filepath = "E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/skZvaGY.jpg"
#import OpenCV
import cv2
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
cv2.startWindowThread()
img = cv2.imread(filepath)
cv2.namedWindow('MessyOrNot', cv2.WINDOW_AUTOSIZE)
cv2.imshow('MessyOrNot',img)
cv2.waitKey(5000)
cv2.destroyAllWindows()

from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 

img_width, img_height = 224, 224

train_data_dir = "E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/we_data/train"
validation_data_dir = "E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/we_data/test"
path_clean = "E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/we_data/train/clean/"
path_messy = "E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/we_data/train/messy/"
nb_train_samples = 16
nb_validation_samples = 4
epochs = 10
batch_size = 2

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.compile(loss ='binary_crossentropy', 
					optimizer ='rmsprop', 
				metrics =['accuracy']) 

train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
			horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='binary') 

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='binary') 

model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size) 

model.save_weights('E:/NTT DATA/CCE/CCE Codebase/Sentiment Analysis/SentimentAnalysis/model_saved.h5') 

#imagename = '0_qzQUpL-gOOSVKXbi.jpg'

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(filepath, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in result]
from tkinter import messagebox
import os
list_clean = os.listdir(path_clean) # dir is your directory path
number_files_clean = len(list_clean)+1
list_messy = os.listdir(path_messy) # dir is your directory path
number_files_messy = len(list_messy)+1
#messagebox.showinfo("MessyOrNot", "Testing")
if result[0][0] == 1:
    messagebox.showinfo("MessyOrNot", "Messy")
    MsgBox = tk.messagebox.askquestion ('MessyOrNot','Was i right ?')
    if MsgBox == 'yes':
        cv2.imwrite(os.path.join(path_messy , str(number_files_messy) + '.jpg'), img)
        cv2.waitKey(5000)
    if MsgBox == 'no':
        cv2.imwrite(os.path.join(path_clean , str(number_files_clean) + '.jpg'), img)
        cv2.waitKey(5000)
        root.destroy()
else:
    messagebox.showinfo("MessyOrNot", "Clean")
    MsgBox = tk.messagebox.askquestion ('MessyOrNot','Was i right ?')
    if MsgBox == 'yes':
        cv2.imwrite(os.path.join(path_clean , str(number_files_clean) + '.jpg'), img)
        cv2.waitKey(5000)
    if MsgBox == 'no':
        cv2.imwrite(os.path.join(path_messy , str(number_files_messy) + '.jpg'), img)
        cv2.waitKey(5000)
        root.destroy()

