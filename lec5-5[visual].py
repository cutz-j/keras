import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.preprocessing import image
import tensorflow as tf
from keras import backend as K
from keras.applications import VGG16

model = models.load_model("d:/data/catsanddogs2.h5")
model.summary()

img_path = "d:/data/dnc/test/cats/cat.1700.jpg"

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) # cnn network 3D tensor
img_tensor /= 255. # normalization

plt.imshow(img_tensor[0])
plt.show()

## 모델 instance ##
layer_outputs = [layer.output for layer in model.layers[:8]] # 하위 8개 layer 출력
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

plt.matshow(activations[3][0, :, :, 0], cmap='viridis')

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1] # w filter
    
    size = layer_activation.shape[1] # size
    n_cols = n_features // images_per_row # 팔렛트
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col*images_per_row+row] # feature map 1개
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size: (row+1) * size] = channel_image # palette
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()       

model = VGG16(weights='imagenet', include_top=False)
model.summary()

layer_name = 'block3_conv1' # layer3 - 1
filter_index = 0

layer_output = model.get_layer(layer_name).output # conv 통과 후 나온 직후 데이터

loss = K.mean(layer_output[:, :, :, filter_index]) # feature map을 loss function으로 학습
grads = K.gradients(loss, model.input)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # l2 norm

## 계산 ##
iterate = K.function([model.input], [loss, grads])
loss_val, grads_val = iterate([np.zeros([1, 150, 150, 3])])

input_img = np.random.random([1, 150, 150, 3]) * 20 + 128. # random gray img --> filter값에 맞게끔 gradient ascent
step = 1. # 일종의 learning_rate
for i in range(40):
    loss_val, grads_val = iterate([input_img])
    input_img += grads_val * step

def deprocessImage(x):
    '''
    function: gradient ascent img를 후처리하여 시각화하는 함수
    
    inputs:
        - x: ascent 출력값
    
    outputs:
        - x: 후처리 이미지
    '''
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1) # 0과 1사이로 클리핑
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8') # rgb pixel 0-255
    return x

def generatePattern(layer_name, filter_index, size=150):
    '''
    function: 각 conv layer filter를 최대화하는 image생성 함수
    
    inputs:
        - layer_name: 해당 layer
        - filter_index: 최대화하고자하는 filter idx
    
    outputs:
        - deprocessImage(img): img
    '''
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_val, grads_val = iterate([input_img])
        input_img += grads_val * step
    img = input_img[0]
    return deprocessImage(img)
plt.imshow(generatePattern('block3_conv1', 0))

def layerVis(layer_name, size=64):
    margin = 5 # filter마다 시각화 공간을 만들기 위한 검정색 마진
    res = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype='uint8')
    for i in range(8):
        for j in range(8):
            filter_img = generatePattern(layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            res[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img # 결과를 각 저장
            
    plt.figure(figsize=(20, 20))
    plt.imshow(res)

for i in range(3, 5):
    layerVis(layer_name='block%d_conv2' %(i))





