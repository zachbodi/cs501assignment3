import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os 

DATASET_PATH  = 'res/photos/train'
TEST_DIR =  'res/photos/test'
IMAGE_SIZE    = (256,256)
NUM_CLASSES   = 5
BATCH_SIZE    = 10  
LEARNING_RATE = 0.001 

train_dataset = image_dataset_from_directory(directory='res/photos/train', label_mode='categorical', class_names = ['anastasiia','michelle','shelli','zach','other'])
test_dataset = image_dataset_from_directory(directory='res/photos/test', label_mode='categorical',class_names = ['anastasiia','michelle','shelli','zach','other'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=50,featurewise_center = True,
                                   featurewise_std_normalization = True,width_shift_range=0.2,
                                   height_shift_range=0.2,shear_range=0.25,zoom_range=0.1,
                                   zca_whitening = True,channel_shift_range = 20,
                                   horizontal_flip = True,vertical_flip = True,
                                   validation_split = 0.2,fill_mode='constant')


train_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "training",seed=42,
                                                  class_mode="categorical")

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,class_mode="categorical")

model = Sequential()

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256, 3))

for layer in base_model.layers:
  layer.trainable=False

model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, batch_size=32, epochs=10)

model.evaluate(test_dataset, batch_size=32)

model.save('model')
