### LSTM: keras + tensorflow ###
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

max_features = 10000 # 사용단어
maxlen = 500 # 최대길이
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)


## lstm ##
model = Sequential()
model.add(Embedding(max_features, 32)) # conv1d
model.add(LSTM(32)) # units
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10, batch_size=128,

                    validation_split=0.2)

### tensorflow vanila RNN ###
n_inputs = 3
n_neurons = 5
n_steps = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

seq_length = tf.placeholder(tf.int32, [None])
output_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                        sequence_length=seq_length)

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='d:/data/mnist.npz')






