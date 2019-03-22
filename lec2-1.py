from keras.datasets import mnist
import os
os.chdir("d:/data")
from keras import models # model unit
from keras import layers # layer
from keras.utils import to_categorical # one-hot encoder
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential() # network init
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) # dense --> fully connected layer
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile --> wrapper

train_images = train_images.reshape((60000, 28*28)) # fully connected
train_images = train_images.astype('float32') / 255 # normalization
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_acc)

## tensor practice ##
x = np.array(12)
print(x)
print(x.ndim)



























