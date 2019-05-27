### numpy RNN 구현 ###
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features)) # 난수
state_t = np.zeros((output_features,)) # 0 초기화

W = np.random.random((output_features, input_features)) # W 
U = np.random.random((output_features, output_features)) # U
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) * np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t # state updates

final_output = np.stack(successive_outputs, axis=0)

