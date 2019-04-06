import numpy as np
import string
from keras.datasets import imdb
from keras import preprocessing

## word one-hot ##
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1 # 고유 인덱스 할당

max_length = 10
res = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        res[i, j, index] = 1
        
## letter one-hot ##
char = string.printable
token_idx = dict(zip(char, range(1, len(char) + 1)))

max_len = 50
res2 = np.zeros((len(samples), max_len, max(token_index.values())+1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        res2[i, j, index] = 1.
        
### embedding ###
max_features = 1000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)































