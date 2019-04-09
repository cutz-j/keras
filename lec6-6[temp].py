import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, optimizers, models
import os
from keras.datasets import imdb
from keras.preprocessing import sequence

data_dir = "d:/data/datasets/jena_climate"
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    '''
    function: batch 처리기
    
    inputs:
        - data: sequence 데이터 --> 10분간격
        - lookback: timestep
        - delay: 타깃(미래스텝)
        - min_index: timestep 범위 지정
        - max_index: timestep 범위 지정
        - shuffle: 섞기
        - batch_size: batch
        - step: 간격
    
    outputs:
        - samples: batch sample
        - targets: 미래예측
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
        
## dataset ##
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=0, max_index=200000, shuffle=True,
                      step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay,
                    min_index=200001, max_index=300000,
                    step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay,
                    min_index=300001, max_index=None,
                    step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

def evaluate_naive_method():
    '''
    function: 상식 수준의 시계열 모델 평가
    '''
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()

def visualize(history):
    loss= history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo')
    plt.plot(epochs, val_loss, 'b')
    plt.show()


## simple model ##
model = models.Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.adam(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

## rnn model ##
model = models.Sequential()
model.add(layers.GRU(units=32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

visualize(history)
## rnn dropout model ##
model = models.Sequential()
model.add(layers.GRU(units=32, dropout=0.2, recurrent_dropout=0.2,
                     input_shape=[None, float_data.shape[-1]]))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)

## rnn deeper model ##
model = models.Sequential()
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
                     return_sequences=True, input_shape=[None, float_data.shape[-1]]))
model.add(layers.GRU(64, acitvation='relu', droput=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)

### bidirectional RNN ###
## sequence reverse ##
max_features = 10000
maxlen = 500

# data preprocessing #
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

## Bi-GRU ##
model = models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.5), 
                               input_shape=[None, float_data.shape[-1]]))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)


## conv1d ##
model = models.Sequential()
model.add(layers.Embedding(max_features, output_dim=128, input_length=maxlen))
model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=optimizers.rmsprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

## 1dconv --> RNN ##
step = 3
lookback = 1440
delay = 144

train_gen = generator(float_data, lookback, delay, 0, 200000, True, step)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, step=step)
test_gen = generator(float_data, lookback=lookback, delay=delay, 300001, None, step)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

model = models.Sequential()
model.add(layers.Conv1D(32, 5, activation='relu', input_shape=[None, float_data.shape[-1]]))
model.add(layers.MaxPooling1D(3))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='rmsprop', loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)







