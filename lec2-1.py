from keras.datasets import mnist
import os
os.chdir("d:/data")
from keras import models
from keras import layers



(train_images, train_labels), (test_images, test_lables) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
