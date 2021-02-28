from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

train_dataset = image_dataset_from_directory(directory='data/train', label_mode='binary')
test_dataset = image_dataset_from_directory(directory='data/test', label_mode='binary')

model = Sequential()

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
  layer.trainable=False

model.add(base_model)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, batch_size=32, epochs=1)

model.evaluate(test_dataset, batch_size=32)

model.save('model')
