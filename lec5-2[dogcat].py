### dog-vs-cat image classification ###
import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator # image preprocessing
import matplotlib.pyplot as plt
from keras.applications import VGG16

## dataset ##
origin_dir = "d:/data/dogs-vs-cats/train"
base_dir = "d:/data/dnc"
os.mkdir(base_dir)

# dataset 분할 #
train_dir = os.path.join(base_dir, "train")
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

## image copy --> dir ##
def imageCopy(origin_src, dest, name='cat', range_num=(0, 1000)):
    fnames = [name+'.{}.jpg'.format(i) for i in range(range_num[0], range_num[1])]
    for fname in fnames:
        src = os.path.join(origin_src, fname)
        dst = os.path.join(dest, fname)
        shutil.copyfile(src, dst)

imageCopy(origin_dir, validation_cats_dir, name='cat', range_num=(1000, 1500))
imageCopy(origin_dir, test_cats_dir, name='cat', range_num=(1500, 2000))
imageCopy(origin_dir, train_dogs_dir, name='dog', range_num=(0, 1000))
imageCopy(origin_dir, validation_dogs_dir, name='dog', range_num=(1000, 1500))
imageCopy(origin_dir, test_dogs_dir, name='dog', range_num=(1500, 2000))

## keras model building ##
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout((0.5)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizers.adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

## preprocessing ##
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


## train ##
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)


## Data Augmentation ##
train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
# rotation: 0~180 #
# width/height: 상하좌우 range #
# shear_range: 전단변환(y축) #
# zoom_range: zoom #
# horizontal_flip: 수평뒤집기 (인물, 풍경) #
# flip_mode: nearest --> 새로 생성된 픽셀 채우는 전략 #
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
model.save('dog_cat.h5')

## 시각화 ##
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo')
plt.plot(epochs, val_acc, 'b')
plt.figure()
plt.plot(epochs, loss, 'bo')
plt.plot(epochs, val_loss, 'b')
plt.show()

### VGG16: feature extraction ###
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
'''
parameter:
    - include_top: 최상위 FC layer를 그대로 살릴 것인지 아닌지, True라면 input_shape = (224, 224, 3) fixed
    - input_shape: include_top에 따라 임의
'''
conv_base.summary() # final_output: (4, 4, 512)

datagen = ImageDataGenerator(rescale=1./255) # not augmentation
batch_size = 20

def extract_features(directory, sample_count):
    '''
    function: pre-trained model에서 FC레이어를 제외한 결과 얻기
    
    inputs:
        - directory: data dir
        - sample_count: data length
    
    outputs:
        - features: al-layer를 통과한 뒤 나오는 array
        - labels: 통과 뒤 나오는 label 확률값
    '''
    features = np.zeros(shape=(sample_count, 4, 4, 512)) # final output
    labels = np.zeros(shape=(sample_count)) # label layer <-- dataset
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size, class_mode='binary') # data generator
    i = 0
    for inputs_batch, labels_batch in generator:
        if i * batch_size > sample_count: # batch가 data 넘어갈 경우
            break
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i+1) * batch_size] = features_batch
        labels[i * batch_size: (i+1) * batch_size] = labels_batch
        i += 1
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4* 4* 512))
test_features = np.reshape(test_features, (1000, 4*4*512)) 

## keras building ##
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.adam(),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30, batch_size=20, validation_data=(validation_features, validation_labels))

i = 0
generator = datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
for inputs_batch, labels_batch in generator:
    if i == 1:
        break
    features_batch = conv_base.predict(inputs_batch)
    i += 1
    
features = np.zeros(shape=[2000, 4, 4, 512])
features[0:20] = features_batch












