import os
os.chdir("d:/data")
from keras.datasets import imdb # 영화review data
import pandas as pd
import numpy as np
from keras import models
from keras import layers
import tensorflow as tf


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000) # dataset
## num_words --> 빈번단어 1000개만 사용 ##

wordIdx = imdb.get_word_index() # word index download
reverseWorkIdx = dict([(value, key) for (key, value) in wordIdx.items()]) # 숫자idx를 key로
decodedReview = ' '.join([reverseWorkIdx.get(i - 3, '?') for i in train_data[0]]) # 첫번째 데이터 문장화

def oneHotSeq(sequences, dimension=10000):
    '''
    function:
        - 데이터 1개당 list로 이뤄진 단어 인덱스를 one-hot matrix 변환
    
    inputs:
        - sequences: 리뷰 1개가 리스트로 이뤄진 2중 리스트
        - dimension: 단어의 차원
    
    output:
        - result: one-hot matrix
    '''
    res = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        res[i, seq] = 1
    return res

x_train = oneHotSeq(train_data)
x_test = oneHotSeq(test_data)


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

### keras model building ###
model = models.Sequential() # building
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # first 16, relu
model.add(layers.Dense(16, activation='relu')) # second 16, relu
model.add(layers.Dense(1, activation='sigmoid')) # final 1, sigmoid
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


### tf building ###
X = tf.placeholder(dtype=tf.float32, shape=(None, 10000))













