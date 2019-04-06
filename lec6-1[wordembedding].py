import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
### 워드 임베딩 ###

## corpus ##
# 10개의 문장 수집 -> 워드 2 벡터 생성 #
corpus =['boy is a young man', 'girl is a young woman', 'queen is a wise woman',
         'king is a strong man', 'princess is a young queen', 'prince is a young king',
         'woman is pretty', 'man is strong', 'princess is a girl will be queen',
         'prince is a boy will be king']

def remove_stop_words(corpus):
    # it's -> it is, #
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for word in stop_words:
            if word in tmp:
                tmp.remove(word)
        results.append(" ".join(tmp))
    return results
        
corpus = remove_stop_words(corpus)

words = []
for text in corpus:
    for word in text.split():
        words.append(word)
words = set(words)

word2int = {}
for i, word in enumerate(words):
    word2int[word] = i

sentences = [s.split() for s in corpus]

## skip-gram, window size:2 ##
WINDOW_SIZE = 2
"""
input: ['boy' 'young' 'man']
output:
xdata ydata
boy young
boy man
young boy
young man
man boy
man young
"""
data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx-WINDOW_SIZE, 0):min(idx+WINDOW_SIZE, len(sentence))+1]:
            if neighbor != word:
                data.append([word, neighbor])
    
word_df = pd.DataFrame(data, columns=["input", "label"])
    
ONE_HOT_DIM = len(words) # one_hot_set len

def encoding(data_idx):
    ## one_hot_encoding ##
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_idx] = 1
    return one_hot_encoding
    
X = []
Y = []

## one-hot 적용 ##
for x, y in zip(word_df['input'], word_df['label']):
    X.append(encoding(word2int[x]))
    Y.append(encoding(word2int[y]))
    
x_train = np.asarray(X)
y_train = np.asarray(Y)

x = tf.placeholder(dtype=tf.float32, shape=[None, ONE_HOT_DIM])
y_label = tf.placeholder(dtype=tf.float32, shape=[None, ONE_HOT_DIM])

EMBEDDING_DIM = 2 ## 결과물의 차원

# hidden layer # --> word vector
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
hidden_layer = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([ONE_HOT_DIM]))
hypothesis = tf.nn.softmax(tf.matmul(hidden_layer, W2) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

iteration = 20000
for i in range(iteration):
    sess.run(train, feed_dict={x:x_train, y_label:y_train})
    if i % 3000 == 0:
        print('iter' + str(i) + 'cost is: ', sess.run(cost, feed_dict={x: x_train, y_label:y_train}))

vectors = sess.run(W1+ b1)
vectors2 = sess.run(W2 + b2)
vectors2 = vectors2.T
vectors_mean = (vectors + vectors2) / 2
df2 = pd.DataFrame(vectors_mean, columns=['x1', 'x2'])
df2['word'] = words

sess.close()
        

fig, ax = plt.subplots()
for word, x1, x2 in zip(df2['word'], df2['x1'], df2['x2']):
    ax.annotate(word, (x1, x2))
padding = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - padding
y_axis_min = np.amin(vectors, axis=0)[1] - padding
x_axis_max = np.amax(vectors, axis=0)[0] + padding
y_axis_max = np.amax(vectors, axis=0)[1] + padding
plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()











