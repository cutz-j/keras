import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers, optimizers, losses
from keras.datasets import reuters
#import os
#os.chdir("d:/data")

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

def vectorizeSeq(sequences, dimension=10000):
    '''
    function: sparse one-hot vector maker
    
    inputs:
        - sequences: work sequences
        - dimension: work frequency
    
    outputs:
        - res: sparse one-hot matrix
    '''
    res = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences): # idx, content
        res[i, sequence] = 1.
    return res

x_train = vectorizeSeq(train_data)
x_test = vectorizeSeq(test_data)

onehot_train_labels = vectorizeSeq(train_labels, dimension=46)
onehot_test_labels = vectorizeSeq(test_labels, dimension=46)

## keras building ##
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,))) # (10000, 64)
model.add(layers.Dense(64, activation='relu')) # (64, 64)
model.add(layers.Dense(46, activation='softmax')) # (64, 46)
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])

## cv-set ##
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = onehot_train_labels[:1000]
partial_y_train = onehot_train_labels[1000:]

## fit ##
history = model.fit(partial_x_train, partial_y_train, 
                    epochs=20, batch_size=512, 
                    validation_data=(x_val, y_val))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro')
plt.plot(epochs, val_loss, 'b-')
plt.show()

res = model.evaluate(x_test, onehot_test_labels)
print(res)

### tf building ###

tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=(None, 10000))
Y = tf.placeholder(dtype=tf.float32, shape=(None, 46))
# layer 1 #
W1 = tf.get_variable('W1', shape=[10000, 64], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([64]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1) # (m,16)

# layer 2 #
W2 = tf.get_variable('W2', shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([64]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2) # m,16

# layer 3 #
W3 = tf.get_variable('W3', shape=[64, 46], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.zeros([46]))
hypothesis = tf.matmul(L2, W3) + b3 # m,1

# cost #
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# session #
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.repeat().batch(512)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess.run(iterator.initializer, feed_dict={X:partial_x_train, Y:partial_y_train})
# train #
for epoch in range(20):
    total_batch = int(partial_x_train.shape[0] / 512)
    train_cost = 0
    for i in range(total_batch): # epoch에 의해 돌아가는 1번 batch 회전
        x_batch, y_batch = sess.run(next_element)
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_batch, Y: y_batch})
        train_cost += cost_val / total_batch
    print("cost: ", train_cost)

# predict #
predicted = tf.equal(tf.argmax(hypothesis, 1),
                     tf.argmax(Y, 1)) # argmax끼리의 비교 (각 데이터 m번째 마다)
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32)) # equal수의 mean --> 정확도

y_hat, acc, cor = sess.run([hypothesis, accuracy, predicted], feed_dict={X: x_test, Y:onehot_test_labels}) # 예측값 run
print(acc)

sess.close()






