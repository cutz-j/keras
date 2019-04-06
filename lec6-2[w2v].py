### w2v ###
import numpy as np
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import os
from keras.preprocessing.text import Tokenizer # token
from keras.preprocessing.sequence import pad_sequences # array화
import tensorflow as tf

max_features = 1000 # 빈번단어 1000개 
maxlen = 20 # 사용 텍스트 길이?

# data download #
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # 영화 리뷰 데이터 --> word index array



# 20차원으로 축소 #
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) # shape=(25000,20)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen) # (25000, 20)

## embedding layer ##
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10, batch_size=32, validation_split=0.2)
model.summary()

### pretrained embedding model ###
imdb_dir = 'd:/data/datasets/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
labels = [] # 25000 label(pos / neg)
texts = [] # 25000 (sentences)

## label 마다 load ##
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
            
maxlen = 100 # 단어개수최대
training_samples = 15000
validation_samples = 10000
max_words = 10000 # dataset에서 사용할 단어개수

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

np.random.seed(7)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

## pretrained: Glove ##
glove_dir = 'd:/data/datasets/'
embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in  f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# embedding matrix #
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# weight set #
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

## tensorflow embedding ##

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

X = tf.placeholder(dtype=tf.float32, shape=[None, 100])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_uniform(shape=[100, 10000], dtype=tf.float32, seed=7))
b1 = tf.Variable(tf.random_uniform(shape=[10000], dtype=tf.float32, seed=7))
flatten = tf.layers.flatten(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform(shape=[10000, 1], dtype=tf.float32, seed=7))
b2 = tf.Variable(tf.random_uniform(shape=[1], dtype=tf.float32, seed=7))
logits = tf.sigmoid(tf.matmul(flatten, W2) + b2)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
train = tf.train.AdamOptimizer(0.1).minimize(cost)

correct = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, correct), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.repeat().batch(32)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer, feed_dict={X:x_train, y:y_train})
    # train #
    for epoch in range(50):
        total_batch = int(x_train.shape[0] / 32)
        train_cost = 0
        for i in range(total_batch): # epoch에 의해 돌아가는 1번 batch 회전
            x_batch, y_batch = sess.run(next_element)
            cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, y: y_batch})
            train_cost += cost_val / total_batch
        print("cost: ", train_cost)
        
    acc, cor, y_hat = sess.run([accuracy, correct, logits],
                                feed_dict={X: x_val, y: y_val})
    print(acc)
    











































