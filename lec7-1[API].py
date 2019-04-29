import numpy as np
import matplotlib.pyplot as plt
from keras import Input, layers, models
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras

voca_size = 50000
num_income_groups = 10

posts_input = Input(shape=[None, ], dtype='int32', name='posts')
embedded_posts = layers.Embedding(voca_size, 256)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = models.Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='adam', 
              loss={'age':'mse', 
                    'income':'categorical_crossentropy', 
                    'gender':'binary_crossentropy'},
              loss_weights={'age':0.25,
                            'income':1.,
                            'gender':10.})

## embedding layers ##
input_arr = np.random.randint(1000, size=(32, 10))
model = models.Sequential()
model.add(layers.Embedding(1000, output_dim=64, input_length=10))
model.compile('rmsprop', 'mse')
res = model.predict(input_arr)
res.shape


### 7.1.2 다중입력모델 ###
text_size = 10000
question_size = 10000
answer_size = 500

text_input = Input(shape=[None,], dtype='int32', name='text')
embedded_text = layers.Embedding(text_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=[None,], dtype='int32', name='question')
embedded_question = layers.Embedding(question_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concat = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_size, activation='softmax')(concat)
model = models.Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

## data 입력 ##
num_samples = 1000
max_length = 100

text = np.random.randint(low=1, high=text_size, size=(num_samples, max_length))
question = np.random.randint(low=1, high=question_size, size=(num_samples, max_length))
answers = np.random.randint(low=0, high=answer_size, size=num_samples)
answers = to_categorical(answers)

model.fit([text, question], answers, epochs=10, batch_size=128)

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
callbacks = [keras.callbacks.TensorBoard(log_dir='d:/data/my_log_dir', histogram_freq=1, embeddings_freq=1,)]
history = model.fit(x_train, y_train,
                    epochs=20, batch_size=128,
                    validation_split=0.2, callbacks=callbacks)



















