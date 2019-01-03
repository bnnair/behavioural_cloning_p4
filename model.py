import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Conv2D
from keras.layers.core import Activation, Reshape
from keras.optimizers import Adam

import math
import cv2
import os

import matplotlib.pyplot as plt
#matplotlib inline

## IMPORT COLUMNS FROM driving_log.csv INTO LISTS ##
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv('driving_log.csv', skiprows=[0], names=colnames)

IMAGE_ROWS = 64
IMAGE_COLS = 64
CHANNELS = 3

# Crop image to remove the sky and driving deck, resize to 64 x 64 dimension 
def crop_resize(image):
    cropped = cv2.resize(image[60:140,:], (IMAGE_ROWS,IMAGE_COLS))
    return cropped

def get_image(filename, resize_crop_flag):
    try:
        image_file = os.path.join('IMG', filename)
        img = cv2.imread(image_file).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize_crop_flag ==True:
            img = crop_resize(img)
        return img
    except IOError:
        print(filename + 'not found')
        return None
    except:
        return None
    
train_samples, validation_samples = train_test_split(data, test_size=0.2)
print(train_samples.shape)
print(validation_samples.shape)


# flip image and steering for half of samples
def flip_image_and_steering(image, steering):
    random = np.random.randint(2)
    if (random == 0):
        image = np.fliplr(image)
        steering = -steering
    return image, steering

def get_steering_angles(samples):
    angles = []
    dummy_image = np.zeros((10,10,3), np.uint8)
    cnt =0
    #print(samples.shape)
    for sample in samples.loc[:,'steering']:
        steering = float(sample) 
        steering_temp = steering
        if flip_active==True:
            image, steering = flip_image_and_steering(dummy_image, steering)
            if (steering_temp!=steering):
                cnt = cnt+1
        angles.append(steering)
    print("{} flips made".format(cnt))    
    return angles


def return_true_for_given_percentage(percent):
    random = np.random.randint(100)
    if random < percent:
        return True
    else:
        return False

def to_float_func(a):
    return float(a)

# Deletes given percentage of samples with steering angle < angle_thresh
def remove_low_steering(data, percentage, angle_thresh):
    array = np.asarray(data)
    vfunc = np.vectorize(to_float_func)
    steer_floats_arr = vfunc(array[:,3])
    #print(steer_floats_arr)
    indexes_small_steer_floats = np.where(np.abs(steer_floats_arr) < angle_thresh)[0]
    rows = []
    for i in list(indexes_small_steer_floats):
        if (return_true_for_given_percentage(percentage)):
            rows.append(i)
    print("{} samples with low steering angle were removed".format(len(rows)))
    data2 = np.delete(array, rows, 0)
    return data2

train_samples = remove_low_steering(train_samples, 80, 0.03)  
print("Number of training samples after removing low steering: {}".format(len(train_samples)))


flip_active = True

# returns indexes for center, left and right camera - each with 33% probability
def get_camera_index():
    random = np.random.randint(3)
    if (random == 0):
        return 0    # center camera index
    elif (random == 0):
        return 1    # left camera index
    else:
        return 2 # right camera index
    
def generator(samples, batch_size, angle_correction):
    num_samples = len(samples)
    #print('num_samples--',num_samples)
    #num_samples = num_samples - (num_samples % 5 )
    #print('num_samples--',num_samples)
    angle_corrections = [0, angle_correction, -angle_correction]
    
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                camera_index = get_camera_index()
                filename = batch_sample[camera_index] #np.int32(camera_index)
                filename = filename.replace('\\','//')
                #print(filename)
                image = get_image(os.path.basename(filename), True)                
                stangle = batch_sample[3]
                #if stangle is not int:
                #    stangle = 0.0
                steering = float(stangle) + angle_corrections[camera_index]
                image, steering = flip_image_and_steering(image, steering)
                #print(image.shape)
                images.append(image)
                angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 
            
            
ANGLE_CORRECTION = 0.25
BATCH_SIZE = 32
NO_EPOCHS = 10

train_generator = generator(train_samples, BATCH_SIZE, ANGLE_CORRECTION)
validation_generator = generator(np.array(validation_samples), BATCH_SIZE, ANGLE_CORRECTION)

def build_model(dropout=.4):
    model = Sequential()
    
    input_shape_after_crop=(IMAGE_ROWS, IMAGE_COLS, CHANNELS)
    # pixels normalization using Lambda method
    model.add(Lambda(lambda x: x/128-0.5, input_shape=input_shape_after_crop))
    
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Flatten())    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse', metrics=['accuracy'])
    model.summary()
    return model


model = build_model()

model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples)/BATCH_SIZE), \
                                     epochs=NO_EPOCHS, validation_data = validation_generator, \
                                     validation_steps = int(len(validation_samples)/BATCH_SIZE))

model.save('model.h5')

