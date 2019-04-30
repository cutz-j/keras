import keras
import numpy as np
from keras import layers
import random
import sys

path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i+maxlen])
    next_chars.append(text[i + maxlen])

chars = sorted(list(set(text)))
char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for j, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[j, t, char_indices[char]] = 1
    y[j, char_indices[next_chars[j]]] = 1
    

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

random.seed(42)
start_index = random.randint(0, len(text) - maxlen - 1)
for epoch in range(1, 60):
    model.fit(x, y, batch_size=128, epochs=1)
    seed_text = text[start_index: start_index + maxlen]
    for temp in [0.2, 0.5, 1.0, 1.2]:
        generated_text = seed_text
        sys.stdout.write(generated_text)
        
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temp)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
















