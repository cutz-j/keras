import os
os.chdir("d:/data")
from keras.datasets import imdb # 영화review data
import pandas as pd
import numpy as np
from keras import models, optimizers, losses, metrics
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt


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


y_train = np.asarray(train_labels).astype('float32').reshape(25000, 1)
y_test = np.asarray(test_labels).astype('float32').reshape(25000, 1)

### keras model building ###
model = models.Sequential() # building
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # first 16, relu
model.add(layers.Dense(16, activation='relu')) # second 16, relu
model.add(layers.Dense(1, activation='sigmoid')) # final 1, sigmoid
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
model.fit(x_train, y_train, batch_size=512, epochs=4)
res = model.evaluate(x_test, y_test)
print(res)


### tf building ###
tf.set_random_seed(777)
tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=(None, 10000))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
# layer 1 #
W1 = tf.get_variable('W1', shape=[10000, 16], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([16]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1) # (m,16)

# layer 2 #
W2 = tf.get_variable('W2', shape=[16, 16], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([16]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2) # m,16

# layer 3 #
W3 = tf.get_variable('W3', shape=[16, 1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([1]))
hypothesis = tf.matmul(L2, W3) + b3 # no need to use sigmoid

# cost #
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# session #
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.repeat().batch(50)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess.run(iterator.initializer, feed_dict={X:x_train, y:y_train})
# train #
for epoch in range(4):
    total_batch = int(x_train.shape[0] / 50)
    train_cost = 0
    for i in range(total_batch): # epoch에 의해 돌아가는 1번 batch 회전
        x_batch, y_batch = sess.run(next_element)
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_batch, y: y_batch})
        train_cost += cost_val / total_batch
    print("cost: ", train_cost)

# predict #
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 임계치 0.5
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32)) # equal수의 mean --> 정확도

y_hat, acc, cor = sess.run([hypothesis, accuracy, predicted], feed_dict={X: x_test, y: y_test}) # 예측값 run
print(acc)

sess.close()


### 훈련 검증 ###
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]



model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

## 시각화 ##
history_dict = history.history # 학습정보가 담긴 사전
loss = history_dict['loss'] # loss 기록
val_loss = history_dict['val_loss'] # validation set loss
epochs = range(1, len(loss) + 1) # 총 epochs

plt.plot(epochs, loss, 'ro', label='training')
plt.plot(epochs, val_loss, 'b', label='test')
plt.legend()
plt.show()




















