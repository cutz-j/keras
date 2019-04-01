### fine tuning ###
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.keras import layers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import os

## dataset ##
origin_dir = "d:/data/dogs-vs-cats/train"
base_dir = "d:/data/dnc"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) # pixel (0, 255) --> (0, 1)

# rescale / class num #
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')


model = tf.keras.models.Sequential()

conv = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
conv.summary()


## trainable 조정 ##
conv.trainable = True

set_trainable = False
for layer in conv.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
## fine tune ##
model.add(conv)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['acc'])
history = model.fit_generator(train_generator,
                              steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)