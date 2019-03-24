import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers, optimizers, losses
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler # 표준화스케일링 라이브러리

tf.reset_default_graph()
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

## 표준화 ##
ss = StandardScaler()
x_train_scale = ss.fit_transform(x_train) # parameter set-up
x_test_scale = ss.transform(x_test) # 적용
print(ss.mean_) # 13개 mean
print(ss.var_) # 13개 var

## keras building ##
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(), loss=losses.mse, metrics=['mse'])
    return model

def kFold(x_train, y_train, k=4, num_epochs=500):
    '''
    function: k-fold cross-validation
    
    inputs:
        - k: dataset 분할 수
        - num_epochs: epochs 수
        - x_train: train data
        - x_test: test data
    
    outputs:
        - all_scores: mae list(np)
        - all_mae_history: epoch마다 기록되는 mae (np)
    '''
    num_val = len(x_train) // k # 데이터 분할 개수
#    all_scores = []
    all_mae_history = []
    for i in range(k):
        tf.reset_default_graph()
        val_data = x_train[i * num_val: (i + 1) * num_val]
        val_target = y_train[i * num_val: (i + 1) * num_val]
        
        partial_train_data = np.concatenate([x_train[:i * num_val], x_train[(i + 1) * num_val:]], axis=0)
        partial_train_targets = np.concatenate([y_train[:i * num_val], y_train[(i + 1) * num_val:]], axis=0)
        
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_target), 
                            epochs=num_epochs, batch_size=1, verbose=1)
#        val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
        mae_history = history.history['val_mean_squared_error']
#        all_scores.append(val_mae)
        all_mae_history.append(mae_history)
    return all_mae_history


all_mae_history = kFold(x_train_scale, y_train)
all_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(500)]    

def smooth_curve(points, factor=0.9):
    '''
    fuction: 지수 이동 평균을 이용해 avg cost 완화
    
    inputs:
        - points: mae score
        - factor: 지수가중평균 parameter
    
    outputs:
        - smooth: 변경된 array
    '''
    smooth = []
    for point in points:
        if smooth:
            previous = smooth[-1]
            smooth.append(previous * factor + point * (1 - factor))
        else:
            smooth.append(point)
    return smooth
    
smooth_history = smooth_curve(all_mae_history)
plt.plot(range(1, len(smooth_history) + 1), smooth_history)
plt.show()    
    
    
## tf building ##
tf.reset_default_graph()
def build_tf_model(x_train, y_train, x_val, y_val):
    '''
    function: x_train, y_train 학습 후, dev-set의 cost 반환 함수
    
    inputs:
        - x_train: x train
        - y_train: y train
        - x_val: x_validation-set
        - y_val: y_validation_set
     
    outputs:
        - val_score: validation cost
    '''
    val_score = []
    X = tf.placeholder(dtype=tf.float32, shape=[None, 13])
    y = tf.placeholder(dtype=tf.float32, shape=[None,])
    
    W1 = tf.get_variable('W1', shape=[13, 64], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([64]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    
    W2 = tf.get_variable('W2', shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([64]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    
    W3 = tf.get_variable('W3', shape=[64, 1], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.zeros([1]))
    hypothesis = tf.matmul(L2, W3) + b3
    
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(500):
            _ = sess.run(optimizer, feed_dict={X: x_train, y: y_train})
            cost_val = sess.run(cost, feed_dict={X: x_val, y: y_val})
            if 500 % 100 == 0:
                print(cost_val)
            val_score.append(cost_val)
    
    return val_score
        
def kFold_tf(x_train, y_train, k=4, num_epochs=500):
    '''
    function: k-fold cross-validation
    
    inputs:
        - k: dataset 분할 수
        - num_epochs: epochs 수
        - x_train: train data
        - x_test: test data
    
    outputs:
        - all_scores: mae list(np)
        - all_mae_history: epoch마다 기록되는 mae (np)
    '''
    num_val = len(x_train) // k # 데이터 분할 개수
    all_scores = []
    for i in range(k):
        tf.reset_default_graph()
        val_data = x_train[i * num_val: (i + 1) * num_val]
        val_target = y_train[i * num_val: (i + 1) * num_val]
        
        partial_train_data = np.concatenate([x_train[:i * num_val], x_train[(i + 1) * num_val:]], axis=0)
        partial_train_targets = np.concatenate([y_train[:i * num_val], y_train[(i + 1) * num_val:]], axis=0)
        
        mse_history = build_tf_model(partial_train_data, partial_train_targets, val_data, val_target)
        all_scores.append(mse_history)

    return all_scores 
    
all_mse_history = kFold_tf(x_train_scale, y_train)
all_mse_history = [np.mean([x[i] for x in all_mse_history]) for i in range(500)]
tf_smooth_history = smooth_curve(all_mae_history)
plt.plot(range(1, len(tf_smooth_history) + 1), tf_smooth_history)
plt.show()     
    
    