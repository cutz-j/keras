from keras.preprocessing import image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt

img_path = "d:/data/creative_commons_elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # 정규화

model = VGG16(weights='imagenet')
preds = model.predict(x)

arg_cls = decode_predictions(preds, top=3)

a_elephant = model.output[:, 386] # 아프리카 코끼리
last_conv_layer = model.get_layer('block5_conv3')  # 마지막레이어 마지막 conv3
grads = K.gradients(a_elephant, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_val, conv_layer_output_val = iterate([x])
for i in range(512):
    conv_layer_output_val[:, :, i] *= pooled_grads_val[i]

heatmap = np.mean(conv_layer_output_val, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)