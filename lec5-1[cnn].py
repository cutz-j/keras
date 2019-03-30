from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

### CNN: mnist ###
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#print(model.summary())

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


## dataset ##
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) # 3d vector 변환
train_images = train_images.astype('float32') / 255 # norm

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels) # one-hot
test_labels = to_categorical(test_labels)

## model ##
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
model.evaluate(test_images, test_labels)

### tf building ###
X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
Y = tf.placeholder(tf.float32, shape=(None, 10)) # classes

#W1 = tf.get_variable('W1', shape=(3, 3, 3, 32), initializer=tf.contrib.layers.xavier_initializer_conv2d())

conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='valid', strides=2)
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding='valid', strides=2)

conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
flat = tf.reshape(conv3, [-1, 64 * conv3.get_shape().as_list()[1] * conv3.get_shape().as_list()[2]])
dense4 = tf.layers.dense(inputs=flat, units=64, activation=tf.nn.relu)
dense5 = tf.layers.dense(inputs=dense4, units=10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense5, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct = tf.equal(tf.argmax(dense5, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.repeat().batch(64)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess.run(iterator.initializer, feed_dict={X:train_images, Y:train_labels})

for epoch in range(5):
    total_batch = int(train_images.shape[0] / 512)
    train_cost = 0
    for i in range(total_batch): # epoch에 의해 돌아가는 1번 batch 회전
        x_batch, y_batch = sess.run(next_element)
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
        train_cost += cost_val / total_batch
    print("cost: ", train_cost)
    
y_hat, acc, cor = sess.run([dense5, accuracy, correct], feed_dict={X: test_images, Y:test_labels}) # 예측값 run
print(acc)
sess.close()









