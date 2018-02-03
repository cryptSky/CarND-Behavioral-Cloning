import csv
import cv2
import numpy as np
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import Dropout, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.layers import Cropping2D
import keras.backend as K
import sklearn
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

samples = []
csv_files = ['./driving_log_7.csv']

for csv_file in csv_files:
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def preprocess_image(image):
    image = image[...,::-1]
    #print(image.shape)
    #image = image[50:-20,:,:]
    #print(crop_img.shape)
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    # the ratio of the new image to the old image
    #r = 100.0 / image.shape[1]
    #dim = (100, int(image.shape[0] * r))
    
    # perform the actual resizing of the image and show it
    #resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    #print(image.shape)
    return image
    
    

    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
    
                # create adjusted steering measurements for the side camera images
                correction = 0.35 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
    
                # read in images from center, left and right cameras
                img_center = np.asarray(preprocess_image(cv2.imread(batch_sample[0])))
                img_left = np.asarray(preprocess_image(cv2.imread(batch_sample[1])))
                img_right = np.asarray(preprocess_image(cv2.imread(batch_sample[2])))
    
                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                
                images.append(np.fliplr(img_center))
                images.append(np.fliplr(img_left))
                images.append(np.fliplr(img_right))
                
                angles.append(-steering_center)
                angles.append(-steering_left)
                angles.append(-steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def get_all_data(batch_samples):
 
    images = []
    angles = []
    
    index = 0
    
    size = len(batch_samples)*6
    images = np.empty((size, 160, 320, 3))
    angles = np.zeros(size)
    
    for batch_sample in batch_samples:
        steering_center = float(batch_sample[3])
    
        # create adjusted steering measurements for the side camera images
        correction = 0.25 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
    
        # read in images from center, left and right cameras
        img_center = np.asarray(preprocess_image(cv2.imread(batch_sample[0])))
        img_left = np.asarray(preprocess_image(cv2.imread(batch_sample[1])))
        img_right = np.asarray(preprocess_image(cv2.imread(batch_sample[2])))
    
        images[index] = img_center
        images[index + 1] = img_left
        images[index + 2] = img_right
        
        angles[index] = steering_center
        angles[index + 1] = steering_left
        angles[index + 2] = steering_right
        
        images[index + 3] = np.fliplr(img_center)
        images[index + 4] = np.fliplr(img_left)
        images[index + 5] = np.fliplr(img_right)
        
        angles[index + 3] = -steering_center
        angles[index + 4] = -steering_left
        angles[index + 5] = -steering_right
        
    return images, angles
    
def visualize_model(model):
    ### print the keys contained in the history object
    print(model.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#X_train, y_train = get_all_data(samples)

def model():
    row, col, ch = 160,320,3  # Trimmed image format

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda x: cv2.resize(x, (0,0), fx=0.5, fy=0.5), input_shape=(90, 320, 3), output_shape=(row, col, ch)))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Conv2D(24, 5, strides=2, padding='same', input_shape=(ch, row, col)))
    model.add(Activation('relu'))
    model.add(Conv2D(36, 5, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(48, 5, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(1))
        
    return model

  
def run(model):
    model.compile(loss='mse', optimizer=Adam())
    
    # checkpoint
    #filepath="weights.best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #callbacks_list = [checkpoint]
    
    #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)      
    #model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
    h = model.fit_generator(train_generator, steps_per_epoch=len(train_samples) // 32, validation_data=validation_generator, validation_steps=len(validation_samples) // 32, max_queue_size=10,
                    callbacks=callbacks_list, epochs=2, verbose = 1)
    
    model.save('model.h5')
    
    visualize_model(h)

   
    
#model = load_model('model.h5')

model = model()
print(model.summary())

run(model)