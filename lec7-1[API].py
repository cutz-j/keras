import numpy as np
import matplotlib.pyplot as plt
from keras import Input, layers, models

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